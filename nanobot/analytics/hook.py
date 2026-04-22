"""PostHog analytics hook for nanobot's agent loop.

Fires: message_sent, agent_responded, feedback_submitted, user_onboarded,
household_member_added.

All calls are fire-and-forget. Classification runs as an async background task
so it never blocks the user response.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.analytics.feedback import detect_feedback
from nanobot.analytics.identity import (
    get_distinct_id,
    get_household_id,
    migrate_channel_hashes,
)

logger = logging.getLogger(__name__)

# How many seconds between messages to count as a followup
_FOLLOWUP_WINDOW_S = 300  # 5 minutes

# Persisted-state file name under the instance analytics data dir.
_STATE_FILENAME = "seen_users.json"
_STATE_VERSION = 1


def _resolve_state_path() -> Path | None:
    """Return the path where onboarding state should live, or None if the
    runtime config isn't available (unit tests, CLI without a config).

    Main and guest nanobots run in separate processes and each call
    `set_config_path()` with their own config file. We namespace under the
    config filename's stem so they don't race on a shared state file —
    e.g. `~/.nanobot/analytics/config/seen_users.json` for main,
    `~/.nanobot/analytics/guest_config/seen_users.json` for guest.
    """
    override = os.environ.get("HOMER_ANALYTICS_STATE_DIR", "").strip()
    if override:
        try:
            path = Path(override).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            return path / _STATE_FILENAME
        except OSError:
            logger.debug("HOMER_ANALYTICS_STATE_DIR unwritable: %s", override)
            return None
    try:
        from nanobot.config.loader import get_config_path
        from nanobot.config.paths import get_runtime_subdir
        subdir = get_runtime_subdir("analytics") / get_config_path().stem
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / _STATE_FILENAME
    except (ImportError, RuntimeError, OSError):
        # No config loaded yet — defer persistence to a later call.
        return None


class AnalyticsHook:
    """Non-blocking PostHog instrumentation wired into AgentLoop._process_message.

    State (seen_users, first_user_ts) is persisted per-process: the on-disk
    path is namespaced by `get_config_path().stem`, so main and guest nanobot
    processes — which share a parent data dir — don't race on the same file.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._initialized = False
        # Track last message timestamp per distinct_id for is_followup
        self._last_message: dict[str, float] = {}
        # Track seen distinct_ids for user_onboarded / household_member_added.
        # Persisted to disk — see _load_state / _save_state — so a restart
        # doesn't re-fire user_onboarded for users we've already tracked.
        self._seen_users: set[str] = set()
        self._first_user_ts: float | None = None
        self._household_id = ""
        self._state_path: Path | None = None
        self._state_loaded = False

    # ── persistence ──────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load seen_users and first_user_ts from disk. Safe on every call."""
        if self._state_loaded:
            return
        self._state_path = _resolve_state_path()
        if self._state_path is None:
            self._state_loaded = True
            return
        try:
            raw = self._state_path.read_text()
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("state file is not an object")
            users = data.get("seen_users") or []
            self._seen_users = {str(u) for u in users if isinstance(u, str)}
            ts = data.get("first_user_ts")
            self._first_user_ts = float(ts) if isinstance(ts, (int, float)) else None
            logger.debug("Analytics state loaded (%d users)", len(self._seen_users))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Analytics state file unreadable (%s) — starting fresh", exc)
        self._state_loaded = True
        self._migrate_seen_users_to_canonical()

    def _migrate_seen_users_to_canonical(self) -> None:
        """Apply identity-map canonicalization to any existing seen_users
        entries, so a deploy that turns on the map doesn't re-fire
        user_onboarded for every known user."""
        if migrate_channel_hashes(self._seen_users):
            self._save_state()

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        payload = {
            "version": _STATE_VERSION,
            "household_id": self._household_id,
            "seen_users": sorted(self._seen_users),
            "first_user_ts": self._first_user_ts,
        }
        try:
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self._state_path)
        except OSError as exc:
            logger.warning("Failed to persist analytics state: %s", exc)

    # ── init ─────────────────────────────────────────────────────────────

    def _ensure_init(self) -> bool:
        """Lazy-init PostHog client. Returns True if client is live."""
        if self._initialized:
            return self._client is not None
        self._initialized = True
        api_key = os.environ.get("POSTHOG_API_KEY", "").strip()
        host = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com").strip()
        self._household_id = get_household_id()
        # Load persisted onboarding state so restarts don't re-fire
        # user_onboarded / household_member_added for known distinct_ids.
        self._load_state()
        if not api_key:
            logger.debug("POSTHOG_API_KEY not set — analytics disabled")
            return False
        try:
            from posthog import Posthog
            self._client = Posthog(api_key, host=host)
            atexit.register(self._client.shutdown)
            logger.info("PostHog analytics initialized (host=%s)", host)
            return True
        except ImportError:
            logger.warning("posthog package not installed — analytics disabled")
            return False

    def _base_props(self, turn_id: str | None = None) -> dict:
        """Base props attached to every event.

        `turn_id` is a per-turn correlation id — all events fired inside a
        single _process_message call carry the same value so duplicate events
        (caller re-invocation, multiple processes) are inspectable in PostHog.
        """
        props: dict[str, Any] = {}
        hid = self._household_id or get_household_id()
        if hid:
            props["household_id"] = hid
        if turn_id:
            props["turn_id"] = turn_id
        return props

    def on_message_received(
        self,
        *,
        channel: str,
        sender_id: str,
        content: str,
        media: list[str],
        timestamp: datetime,
        is_guest: bool,
    ) -> dict[str, Any]:
        """Call at the start of _process_message. Returns context dict for on_response_sent."""
        return {
            "channel": channel,
            "sender_id": sender_id,
            "content": content,
            "media": media,
            "timestamp": timestamp,
            "is_guest": is_guest,
            "inbound_time": time.monotonic(),
            "turn_id": uuid.uuid4().hex[:16],
        }

    async def on_response_sent(
        self,
        ctx: dict[str, Any],
        *,
        response_content: str | None,
        tools_used: set[str],
        escalation_triggered: bool = False,
        schedule_background: Any = None,
    ) -> None:
        """Call after the response OutboundMessage is built.

        Fires agent_responded immediately. Fires message_sent and feedback_submitted
        as background tasks (classification is async).
        """
        if not self._ensure_init():
            return

        channel = ctx["channel"]
        sender_id = ctx["sender_id"]
        content = ctx["content"]
        turn_id = ctx.get("turn_id") or uuid.uuid4().hex[:16]
        distinct_id = get_distinct_id(sender_id, channel)
        now = time.monotonic()
        latency_ms = int((now - ctx["inbound_time"]) * 1000)

        # ── agent_responded (immediate) ──────────────────────────────────
        self._client.capture(distinct_id, "agent_responded", {
            **self._base_props(turn_id),
            "channel": channel,
            "latency_ms": latency_ms,
            "tool_calls_count": len(tools_used),
            "tools_used": sorted(tools_used),
            "escalation_triggered": escalation_triggered,
            "response_length": len(response_content) if response_content else 0,
        })

        # ── feedback_submitted (immediate, cheap check) ──────────────────
        fb = detect_feedback(content)
        if fb:
            self._client.capture(distinct_id, "feedback_submitted", {
                **self._base_props(turn_id),
                "sentiment": fb.sentiment,
                "trigger": fb.trigger,
            })

        # ── user_onboarded / household_member_added (first-seen check) ───
        self._maybe_fire_onboarding(distinct_id, channel, turn_id)

        # ── message_sent (background — waits for classification) ─────────
        is_followup = self._check_followup(distinct_id, now)

        coro = self._fire_message_sent(
            distinct_id=distinct_id,
            channel=channel,
            content=content,
            media=ctx["media"],
            is_followup=is_followup,
            turn_id=turn_id,
        )
        if schedule_background:
            schedule_background(coro)
        else:
            asyncio.create_task(coro)

    def _check_followup(self, distinct_id: str, now: float) -> bool:
        last = self._last_message.get(distinct_id)
        self._last_message[distinct_id] = now
        if last is None:
            return False
        return (now - last) < _FOLLOWUP_WINDOW_S

    def _maybe_fire_onboarding(
        self, distinct_id: str, channel: str, turn_id: str,
    ) -> None:
        """Fire user_onboarded on first message from a new distinct_id.

        `household_member_added` is intentionally NOT fired here — that
        event is now owned by homer's explicit add-member flow
        (tools/manage_users.py) so it reflects real admin actions, not
        "same human on a new channel got hashed differently." Ditto
        guest_added, which homer emits from the add-event-guest flow.
        """
        if distinct_id in self._seen_users:
            return
        is_new_household = len(self._seen_users) == 0
        self._seen_users.add(distinct_id)

        if is_new_household:
            self._first_user_ts = time.time()

        # Persist before capturing so a crash mid-turn can't cause a
        # re-fire on the next boot.
        self._save_state()

        # Identify person + set properties
        self._client.identify(distinct_id, {
            "household_id": self._household_id,
            "channel_first_seen": channel,
            "is_primary": is_new_household,
            "signup_source": "friends_launch",
        })

        self._client.capture(distinct_id, "user_onboarded", {
            **self._base_props(turn_id),
            "channel": channel,
            "is_new_household": is_new_household,
            "signup_source": "friends_launch",
        })

        # Group identify on every new user
        if self._household_id:
            self._client.group_identify("household", self._household_id, {})

    async def _fire_message_sent(
        self,
        *,
        distinct_id: str,
        channel: str,
        content: str,
        media: list[str],
        is_followup: bool,
        turn_id: str,
    ) -> None:
        """Classify, then fire message_sent."""
        from nanobot.analytics.classify import classify_message_async

        tag = await classify_message_async(content)
        self._client.capture(distinct_id, "message_sent", {
            **self._base_props(turn_id),
            "channel": channel,
            "message_length": len(content),
            "has_attachment": bool(media),
            "use_case_tag": tag,
            "is_followup": is_followup,
        })

        # Group identify on every message_sent
        if self._household_id:
            self._client.group_identify("household", self._household_id, {})


# Module-level singleton
_hook: AnalyticsHook | None = None


def get_analytics_hook() -> AnalyticsHook:
    """Return the module-level analytics hook (lazy-created)."""
    global _hook
    if _hook is None:
        _hook = AnalyticsHook()
    return _hook
