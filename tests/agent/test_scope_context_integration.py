"""End-to-end test for scope-context-provider injection through ``_process_message``.

The unit tests in ``tests/test_scope_context_injection.py`` cover the resolver
(``_get_scope_context``) in isolation. The unit tests in
``tests/test_loop_event_agent.py`` cover the workspace-routing helper
(``_resolve_guest_agent_workspace``) in isolation. Neither catches the gate
that wires them together inside ``_process_message`` — and that gate is what
broke for non-workspace-isolated guests until #71.

This test exercises the actual integration:
- Build an ``AgentLoop`` with a configured ``scope_context_provider``.
- Mock ``_run_agent_loop`` so we can inspect the prompt the agent would have
  sent to the LLM. Mocked side_effect raises so we don't have to mock the
  rest of the loop.
- Send a message via ``_process_message`` and assert on the captured prompt.

Three scenarios:
1. Sender with scope data and NO per-scope-type workspace override
   (Adam-style relationship-scope guest) → scope context must be injected.
2. Sender with scope data AND a per-scope-type workspace override
   (Helen-style historian) → scope context must still be injected.
3. Sender with NO scope data → no system message added, agent runs unchanged.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


# ---------------------------------------------------------------------------
# Fake provider module — registered in sys.modules so the loop's importlib
# lookup finds it. Each test parametrises which sender_id maps to which
# scope context.
# ---------------------------------------------------------------------------


def _install_fake_provider(monkeypatch, sender_to_context: dict[str, str]) -> None:
    """Register ``test_scope_module.render`` returning per-sender canned text."""
    mod = types.ModuleType("test_scope_module")

    def render(sender_id: str) -> str:
        return sender_to_context.get(sender_id, "")

    mod.render = render
    monkeypatch.setitem(sys.modules, "test_scope_module", mod)


def _make_loop(
    tmp_path: Path, *, scope_provider: str = "test_scope_module:render"
) -> AgentLoop:
    """Build a real AgentLoop with the scope_context_provider wired in."""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        scope_context_provider=scope_provider,
    )


def _intercept_agent_loop(loop: AgentLoop) -> dict:
    """Replace ``_run_agent_loop`` with a side-effect mock that raises.

    Captures the ``messages`` argument so the test can inspect what would have
    been sent to the LLM. Returns a dict that gets populated with the captured
    messages on the next call.
    """
    captured: dict = {}

    async def _capture(messages, *args, **kwargs):
        captured["messages"] = messages
        raise RuntimeError("intercepted before LLM call")

    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    loop._run_agent_loop = _capture  # type: ignore[method-assign]
    return captured


def _system_messages(messages: list[dict]) -> list[str]:
    """Extract just the system-role message contents (strings) from a prompt."""
    out: list[str] = []
    for m in messages:
        if m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    out.append(block.get("text", ""))
    return out


# ---------------------------------------------------------------------------
# Scenario 1: relationship-scope guest, no workspace override
# (Adam-style — the regression this PR closes)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_context_injected_for_sender_without_workspace_override(
    tmp_path: Path, monkeypatch
) -> None:
    """A guest with scope data but no per-scope-type subdir mapping must still
    have their scope context injected — pre-#71 this silently no-op'd."""
    sender = "16072348189"
    canned_ctx = "# Scope Context\n\n## Scope: denver_mtb\nProposing April 24-26"
    _install_fake_provider(monkeypatch, {sender: canned_ctx})

    loop = _make_loop(tmp_path)
    captured = _intercept_agent_loop(loop)

    # No scope_workspaces.json in the workspace — _resolve_guest_agent_workspace
    # will return None for this sender. That is the case being fixed.
    msg = InboundMessage(
        channel="whatsapp",
        sender_id=sender,
        chat_id=f"{sender}@s.whatsapp.net",
        content="do we have a date?",
    )
    with pytest.raises(RuntimeError, match="intercepted"):
        await loop._process_message(msg)

    sys_msgs = _system_messages(captured["messages"])
    assert any(canned_ctx in m for m in sys_msgs), (
        f"scope context not in prompt — system messages were: {sys_msgs}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: workspace-isolated guest (Helen-style historian)
# Behaviour must be preserved — scope context still injected.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_context_injected_for_workspace_isolated_sender(
    tmp_path: Path, monkeypatch
) -> None:
    """A guest mapped to a per-scope-type subdir via scope_workspaces.json must
    keep getting scope context injection — this is the historian flow."""
    sender = "14126364194"
    canned_ctx = "# Scope Context\n\n## Scope: family_history\nLineage tasks for Helen"
    _install_fake_provider(monkeypatch, {sender: canned_ctx})

    # Wire up the workspace-override mapping the way homer does.
    subdir = tmp_path / "_family_history"
    subdir.mkdir()
    (tmp_path / "scope_workspaces.json").write_text(
        json.dumps({sender: "_family_history"}), encoding="utf-8"
    )

    loop = _make_loop(tmp_path)
    captured = _intercept_agent_loop(loop)

    msg = InboundMessage(
        channel="whatsapp",
        sender_id=sender,
        chat_id=f"{sender}@s.whatsapp.net",
        content="hi homer",
    )
    with pytest.raises(RuntimeError, match="intercepted"):
        await loop._process_message(msg)

    sys_msgs = _system_messages(captured["messages"])
    assert any(canned_ctx in m for m in sys_msgs), (
        f"scope context not in prompt — system messages were: {sys_msgs}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: sender with no scope data
# Provider returns "" → no system message inserted, prompt unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_scope_context_when_provider_returns_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """When the provider returns empty for a sender, no extra system message
    is added — guards against silent-noise injection on cold-start senders."""
    sender = "99999999"
    _install_fake_provider(monkeypatch, {})  # provider returns "" for everyone

    loop = _make_loop(tmp_path)
    captured = _intercept_agent_loop(loop)

    msg = InboundMessage(
        channel="whatsapp",
        sender_id=sender,
        chat_id=f"{sender}@s.whatsapp.net",
        content="random",
    )
    with pytest.raises(RuntimeError, match="intercepted"):
        await loop._process_message(msg)

    sys_msgs = _system_messages(captured["messages"])
    # Whatever system messages exist, none of them should be a scope-context
    # block (which always starts with "# Scope Context").
    scope_blocks = [m for m in sys_msgs if m.startswith("# Scope Context")]
    assert scope_blocks == [], (
        f"unexpected scope context injected for empty-provider sender: {scope_blocks}"
    )


# ---------------------------------------------------------------------------
# Scenario 4: anonymous sender (no sender_id) — provider must not be called.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_scope_context_when_sender_id_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """When sender_id is empty, the provider should not be consulted at all —
    avoids confusing the LLM with someone else's scope when an unidentified
    message slips through (e.g. system / cron-triggered turns)."""
    calls: list[str] = []

    def render(sender_id: str) -> str:
        calls.append(sender_id)
        return "# Scope Context\n\n## Scope: should_not_appear"

    mod = types.ModuleType("test_no_call_module")
    mod.render = render
    monkeypatch.setitem(sys.modules, "test_no_call_module", mod)

    loop = _make_loop(tmp_path, scope_provider="test_no_call_module:render")
    captured = _intercept_agent_loop(loop)

    msg = InboundMessage(
        channel="cli",
        sender_id="",  # explicitly missing
        chat_id="local",
        content="hello",
    )
    with pytest.raises(RuntimeError, match="intercepted"):
        await loop._process_message(msg)

    assert calls == [], f"provider should not be called for empty sender_id: {calls}"
    sys_msgs = _system_messages(captured["messages"])
    assert not any("should_not_appear" in m for m in sys_msgs)
