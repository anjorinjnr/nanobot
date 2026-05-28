"""WhatsApp connection health monitor.

WhatsApp linked-device sessions drop (transient reconnects), flap (rapid
churn that can trip WhatsApp's anti-abuse), or log out entirely (server
revokes the device — requires manual re-pairing). When that happens the
agent keeps running but messages silently fail to deliver. Operators have
no visibility unless they tail logs.

This monitor turns connection-state transitions into push alerts on a
*different* channel (typically the admin's Telegram, which stays up when
WhatsApp is down) so an outage is visible immediately. It also exposes a
telemetry hook so every transition can be recorded for trend analysis.

Opt-in: with no alert destination configured, the monitor still tracks
state and fires telemetry, but sends no alerts.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from loguru import logger

# Async callback that delivers an alert string to the operator.
AlertSender = Callable[[str], Awaitable[None]]
# Sync callback for telemetry: (prev_status, new_status, duration_in_prev_s).
TelemetryHook = Callable[[str, str, float], None]


class WhatsAppHealthMonitor:
    """Tracks WhatsApp connection-state transitions and emits alerts.

    Wired into ``WhatsAppChannel._on_status``. All timing is injectable so
    the state machine is deterministically testable.
    """

    def __init__(
        self,
        *,
        send_alert: AlertSender | None,
        disconnect_alert_after_s: float = 60.0,
        flap_threshold: int = 3,
        flap_window_s: float = 60.0,
        alert_debounce_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        telemetry: TelemetryHook | None = None,
    ):
        # send_alert=None → monitor runs (telemetry + state) but never pings.
        # This is the opt-in switch: tenants whose admin has no alert
        # channel configured simply pass None.
        self._send_alert = send_alert
        self._disconnect_alert_after_s = disconnect_alert_after_s
        self._flap_threshold = flap_threshold
        self._flap_window_s = flap_window_s
        self._alert_debounce_s = alert_debounce_s
        self._clock = clock
        self._telemetry = telemetry

        self._status: str | None = None
        self._status_since: float = clock()
        self._transitions: deque[float] = deque()
        self._last_alert_at: dict[str, float] = {}
        # True once a "down" alert has gone out for the current outage, so we
        # know to send the all-clear on reconnect. Reset on recovery.
        self._outage_alerted = False
        self._pending_disconnect_task: asyncio.Task | None = None

    async def on_status_change(self, status: str) -> None:
        now = self._clock()
        prev = self._status
        if status == prev:
            return  # de-dupe repeated identical events (neonize emits doubles)

        duration_in_prev = now - self._status_since
        if self._telemetry is not None and prev is not None:
            try:
                self._telemetry(prev, status, duration_in_prev)
            except Exception:
                logger.exception("WA health telemetry hook failed")

        self._status = status
        self._status_since = now
        # Only disconnect-type events count toward flap detection. A clean
        # outage is one disconnect + one reconnect; counting reconnects too
        # would flag every normal recovery as "flapping". Real flapping is
        # many *drops* in a short window.
        if status in ("disconnected", "logged_out"):
            self._transitions.append(now)
            self._trim_transitions(now)

        # Cancel any in-flight "still disconnected?" timer — a new transition
        # supersedes it.
        self._cancel_pending_disconnect()

        if status == "logged_out":
            # Terminal-bad: server revoked the device. Needs re-pairing.
            # Alert immediately.
            await self._maybe_alert(
                "logged_out",
                "⚠️ Homer's WhatsApp logged out — the linked device was "
                "revoked. Messages won't deliver until it's re-paired.",
            )
            self._outage_alerted = True
        elif status == "disconnected":
            # Could be a transient reconnect. Don't cry wolf — wait and
            # re-check. The timer is cancelled if we reconnect or transition
            # again before it fires.
            self._schedule_disconnect_alert()
        elif status == "connected":
            if self._outage_alerted:
                await self._maybe_alert(
                    "recovered",
                    "✅ Homer's WhatsApp reconnected. Back to normal.",
                    force=True,  # all-clear should always go out
                )
                self._outage_alerted = False

        # Independent of the specific status: rapid churn is its own signal.
        if len(self._transitions) >= self._flap_threshold:
            await self._maybe_alert(
                "flap",
                "⚠️ Homer's WhatsApp connection is unstable — "
                f"{len(self._transitions)} reconnects in the last "
                f"{int(self._flap_window_s)}s. Deliveries may be dropping.",
            )

    def _trim_transitions(self, now: float) -> None:
        cutoff = now - self._flap_window_s
        while self._transitions and self._transitions[0] < cutoff:
            self._transitions.popleft()

    def _cancel_pending_disconnect(self) -> None:
        if self._pending_disconnect_task and not self._pending_disconnect_task.done():
            self._pending_disconnect_task.cancel()
        self._pending_disconnect_task = None

    def _schedule_disconnect_alert(self) -> None:
        async def _check() -> None:
            try:
                await asyncio.sleep(self._disconnect_alert_after_s)
            except asyncio.CancelledError:
                return
            # Still disconnected after the grace period → real outage.
            if self._status == "disconnected":
                await self._maybe_alert(
                    "disconnected",
                    "⚠️ Homer's WhatsApp has been disconnected for "
                    f"{int(self._disconnect_alert_after_s)}s and hasn't "
                    "recovered. Messages may not be delivering.",
                )
                self._outage_alerted = True

        self._pending_disconnect_task = asyncio.ensure_future(_check())

    async def _maybe_alert(self, key: str, text: str, *, force: bool = False) -> None:
        if self._send_alert is None:
            return
        now = self._clock()
        last = self._last_alert_at.get(key)
        if not force and last is not None and (now - last) < self._alert_debounce_s:
            return  # debounced
        self._last_alert_at[key] = now
        try:
            await self._send_alert(text)
        except Exception:
            logger.exception("WA health alert send failed (key={})", key)
