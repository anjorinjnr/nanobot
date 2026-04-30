"""Tests for exec tool environment isolation."""

import sys

import pytest

from nanobot.agent.tools.shell import ExecTool

_UNIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="Unix shell commands")


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_does_not_leak_parent_env(monkeypatch):
    """Env vars from the parent process must not be visible to commands."""
    monkeypatch.setenv("NANOBOT_SECRET_TOKEN", "super-secret-value")
    tool = ExecTool()
    result = await tool.execute(command="printenv NANOBOT_SECRET_TOKEN")
    assert "super-secret-value" not in result


@pytest.mark.asyncio
async def test_exec_has_working_path():
    """Basic commands should be available via the login shell's PATH."""
    tool = ExecTool()
    result = await tool.execute(command="echo hello")
    assert "hello" in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_path_append():
    """The pathAppend config should be available in the command's PATH."""
    tool = ExecTool(path_append="/opt/custom/bin")
    result = await tool.execute(command="echo $PATH")
    assert "/opt/custom/bin" in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_path_append_preserves_system_path():
    """pathAppend must not clobber standard system paths."""
    tool = ExecTool(path_append="/opt/custom/bin")
    result = await tool.execute(command="ls /")
    assert "Exit code: 0" in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_allowed_env_keys_passthrough(monkeypatch):
    """Env vars listed in allowed_env_keys should be visible to commands."""
    monkeypatch.setenv("MY_CUSTOM_VAR", "hello-from-config")
    tool = ExecTool(allowed_env_keys=["MY_CUSTOM_VAR"])
    result = await tool.execute(command="printenv MY_CUSTOM_VAR")
    assert "hello-from-config" in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_allowed_env_keys_does_not_leak_others(monkeypatch):
    """Env vars NOT in allowed_env_keys should still be blocked."""
    monkeypatch.setenv("MY_CUSTOM_VAR", "hello-from-config")
    monkeypatch.setenv("MY_SECRET_VAR", "secret-value")
    tool = ExecTool(allowed_env_keys=["MY_CUSTOM_VAR"])
    result = await tool.execute(command="printenv MY_SECRET_VAR")
    assert "secret-value" not in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_allowed_env_keys_missing_var_ignored(monkeypatch):
    """If an allowed key is not set in the parent process, it should be silently skipped."""
    monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
    tool = ExecTool(allowed_env_keys=["NONEXISTENT_VAR_12345"])
    result = await tool.execute(command="printenv NONEXISTENT_VAR_12345")
    assert "Exit code: 1" in result


# ── Sender identity injection ───────────────────────────────────────────────
# When the agent loop calls set_context(), nanobot stamps the verified sender
# identity onto the subprocess env as NANOBOT_SENDER_ID / NANOBOT_SENDER_CHANNEL.
# Trusted scripts (e.g. manage_users.py) authenticate the requester from these
# vars instead of trusting LLM-supplied CLI args.


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_injects_sender_env_when_context_set():
    tool = ExecTool()
    tool.set_context(channel="telegram", chat_id="123", sender_id="user-42")
    result = await tool.execute(command="printenv NANOBOT_SENDER_ID")
    assert "user-42" in result
    result = await tool.execute(command="printenv NANOBOT_SENDER_CHANNEL")
    assert "telegram" in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_omits_sender_env_when_context_unset():
    tool = ExecTool()
    # Default state: no set_context() call.
    result = await tool.execute(command="printenv NANOBOT_SENDER_ID")
    # printenv exits 1 when the var is unset.
    assert "Exit code: 1" in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_sender_env_not_read_from_parent_env(monkeypatch):
    """Stale NANOBOT_SENDER_ID in parent OS env must NOT leak to subprocess.

    This is the spoof-resistance contract: the only source of truth for
    sender identity is set_context(), never os.environ. An attacker who
    somehow influenced the parent env cannot impersonate a sender.
    """
    monkeypatch.setenv("NANOBOT_SENDER_ID", "spoofed-admin")
    monkeypatch.setenv("NANOBOT_SENDER_CHANNEL", "telegram")
    tool = ExecTool()
    # No set_context() called → env vars must not be exported.
    result = await tool.execute(command="printenv NANOBOT_SENDER_ID")
    assert "spoofed-admin" not in result
    assert "Exit code: 1" in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_sender_env_overrides_allowed_env_key(monkeypatch):
    """If a misconfigured allowed_env_keys lists NANOBOT_SENDER_ID, the
    runtime-injected value still wins (and absence still means absence)."""
    monkeypatch.setenv("NANOBOT_SENDER_ID", "spoofed-admin")
    tool = ExecTool(allowed_env_keys=["NANOBOT_SENDER_ID"])
    tool.set_context(channel="telegram", chat_id="1", sender_id="real-user")
    result = await tool.execute(command="printenv NANOBOT_SENDER_ID")
    assert "real-user" in result
    assert "spoofed-admin" not in result


@_UNIX_ONLY
@pytest.mark.asyncio
async def test_exec_set_context_clears_sender_when_none():
    """A subsequent turn with sender_id=None must clear the stamp from the
    previous turn, never carry it over."""
    tool = ExecTool()
    tool.set_context(channel="telegram", chat_id="1", sender_id="user-1")
    tool.set_context(channel="cli", chat_id="direct", sender_id=None)
    result = await tool.execute(command="printenv NANOBOT_SENDER_ID")
    assert "user-1" not in result
    assert "Exit code: 1" in result
