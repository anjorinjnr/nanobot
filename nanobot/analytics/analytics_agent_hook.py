"""AgentHook adapter for the PostHog analytics hook.

Bridges the impl in ``analytics/hook.py`` (which fires ``message_sent`` /
``agent_responded`` / ``feedback_submitted`` / ``user_onboarded`` for user
turns and ``agent_initiated_action`` for synthetic turns) onto the
``AgentHook.before_turn`` / ``after_turn`` lifecycle.

Two paths land in the same adapter, branched on ``turn.is_synthetic``:

- **User turns**: ``before_turn`` calls ``on_message_received`` and stashes
  the resulting ctx; ``after_turn`` calls ``on_response_sent`` with the
  built reply, tools list, and an ``escalation_triggered`` derived from
  whether the agent used ``escalate``/``resolve_escalation``.

- **Synthetic turns** (heartbeat, cron): ``before_turn`` is a no-op (the
  upstream event has no inbound ctx); ``after_turn`` calls
  ``track_agent_initiated_action``. The empty-final placeholder is
  dropped before reporting so ``had_outbound`` stays accurate.

Failure-tolerant — exceptions are logged at debug and never propagate.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, TurnMetadata
from nanobot.agent.runner import STOP_EMPTY_FINAL, STOP_INTENTIONAL_SILENCE
from nanobot.analytics.hook import AnalyticsHook, get_analytics_hook

_STATE_KEY = "analytics"


def _escalation_triggered(tools_used: list[str]) -> bool:
    return "escalate" in tools_used or "resolve_escalation" in tools_used


class AnalyticsAgentHook(AgentHook):
    """AgentHook front-end for AnalyticsHook (PostHog write-through).

    ``schedule_background`` is the AgentLoop's background-task scheduler —
    passed through so the classify-then-emit ``message_sent`` flow runs off
    the response path.
    """

    def __init__(
        self,
        *,
        schedule_background: Any | None = None,
        impl: AnalyticsHook | None = None,
    ) -> None:
        super().__init__()
        self._impl = impl if impl is not None else get_analytics_hook()
        self._schedule_background = schedule_background

    async def before_turn(self, turn: TurnMetadata) -> None:
        if turn.is_synthetic:
            return
        try:
            ctx = self._impl.on_message_received(
                channel=turn.channel,
                sender_id=turn.sender_id,
                content=turn.content,
                media=turn.media,
                timestamp=turn.timestamp,
                is_guest=turn.is_guest,
            )
        except Exception:
            logger.debug("analytics before_turn error (non-fatal)", exc_info=True)
            return
        if ctx is not None:
            turn.state[_STATE_KEY] = ctx

    async def after_turn(self, turn: TurnMetadata) -> None:
        if turn.is_synthetic:
            await self._after_synthetic(turn)
            return
        ctx = turn.state.get(_STATE_KEY)
        if ctx is None:
            return
        try:
            await self._impl.on_response_sent(
                ctx,
                response_content=turn.response_content,
                tools_used=turn.tools_used,
                escalation_triggered=_escalation_triggered(turn.tools_used),
                schedule_background=self._schedule_background,
            )
        except Exception:
            logger.debug("analytics after_turn error (non-fatal)", exc_info=True)

    async def _after_synthetic(self, turn: TurnMetadata) -> None:
        # Drop the empty-final placeholder so had_outbound reflects reality.
        if turn.stop_reason in (STOP_EMPTY_FINAL, STOP_INTENTIONAL_SILENCE):
            final = None
        else:
            final = turn.response_content
        latency_ms = int((time.monotonic() - turn.started_at_monotonic) * 1000)
        try:
            self._impl.track_agent_initiated_action(
                trigger_kind=turn.trigger_kind or "synthetic",
                response_content=final,
                tools_used=turn.tools_used,
                latency_ms=latency_ms,
            )
        except Exception:
            logger.debug("agent_initiated_action emit failed", exc_info=True)
