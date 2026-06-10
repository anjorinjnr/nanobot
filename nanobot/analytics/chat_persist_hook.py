"""AgentHook adapter for ChatPersistHook.

Bridges the impl in ``chat_persist.py`` (channel-agnostic Supabase write-through
for hist_chat_messages) onto the ``AgentHook.before_turn`` / ``after_turn``
lifecycle so it stops being called inline from ``AgentLoop._process_message``.

Behavior is intentionally identical to the previous inline calls:
- ``before_turn``: skip synthetic turns; call ``on_message_received`` and stash
  the returned contributor ctx in ``turn.state`` so ``after_turn`` can pair it
  with the assistant reply.
- ``after_turn``: when ctx is present (channel supported + sender resolved),
  call ``on_response_sent`` with the built reply.

Failure-tolerant — exceptions are logged at debug and never propagate. The
underlying impl already silences Supabase errors at warning level.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

from nanobot.agent.hook import AgentHook, TurnMetadata
from nanobot.analytics.chat_persist import ChatPersistHook, get_chat_persist_hook

_STATE_KEY = "chat_persist"


class ChatPersistAgentHook(AgentHook):
    """AgentHook front-end for the chat_persist write-through.

    ``schedule_background`` is the AgentLoop's background-task scheduler; passed
    through so media uploads (and the assistant insert) run off the response
    path.
    """

    def __init__(
        self,
        *,
        schedule_background: Callable[[Any], None] | None = None,
        impl: ChatPersistHook | None = None,
    ) -> None:
        super().__init__()
        self._impl = impl if impl is not None else get_chat_persist_hook()
        self._schedule_background = schedule_background

    async def before_turn(self, turn: TurnMetadata) -> None:
        if turn.is_synthetic:
            return
        try:
            ctx = await self._impl.on_message_received(
                channel=turn.channel,
                sender_id=turn.sender_id,
                content=turn.content,
                media=turn.media,
                timestamp=turn.timestamp,
                schedule_background=self._schedule_background,
            )
        except Exception:
            logger.debug("chat_persist before_turn error (non-fatal)", exc_info=True)
            return
        if ctx is not None:
            turn.state[_STATE_KEY] = ctx

    async def after_turn(self, turn: TurnMetadata) -> None:
        ctx = turn.state.get(_STATE_KEY)
        if ctx is None:
            return
        try:
            await self._impl.on_response_sent(
                ctx,
                response_content=turn.response_content,
                schedule_background=self._schedule_background,
            )
        except Exception:
            logger.debug("chat_persist after_turn error (non-fatal)", exc_info=True)
