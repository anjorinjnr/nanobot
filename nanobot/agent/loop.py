"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from contextlib import AsyncExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import Consolidator, Dream
from nanobot.agent.runner import (
    STOP_EMPTY_FINAL,
    STOP_ERROR,
    STOP_INTENTIONAL_SILENCE,
    STOP_MAX_ITERATIONS,
    _MAX_INJECTIONS_PER_TURN,
    AgentRunner,
    AgentRunSpec,
)
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.notebook import NotebookEditTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.search import GlobTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.self import MyTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.document import extract_documents
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.progress_events import (
    build_tool_event_finish_payloads,
    build_tool_event_start_payload,
    invoke_on_progress,
    on_progress_accepts_tool_events,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE, is_blank_text

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, ToolsConfig, WebToolsConfig
    from nanobot.cron.service import CronService


UNIFIED_SESSION_KEY = "unified:default"


class _LoopHook(AgentHook):
    """Core hook for the main loop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        sender_id: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._sender_id = sender_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from nanobot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._loop._current_iteration = context.iteration

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            tool_events = [build_tool_event_start_payload(tc) for tc in context.tool_calls]
            await invoke_on_progress(
                self._on_progress,
                tool_hint,
                tool_hint=True,
                tool_events=tool_events,
            )
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(
            self._channel, self._chat_id, self._message_id, sender_id=self._sender_id,
        )

    async def after_iteration(self, context: AgentHookContext) -> None:
        if (
            self._on_progress
            and context.tool_calls
            and context.tool_events
            and on_progress_accepts_tool_events(self._on_progress)
        ):
            tool_events = build_tool_event_finish_payloads(context)
            if tool_events:
                await invoke_on_progress(
                    self._on_progress,
                    "",
                    tool_hint=False,
                    tool_events=tool_events,
                )
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        guest_workspace: Path | None = None,
        session_ttl_minutes: int = 0,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        scope_context_provider: str = "",
        disable_memory_writes: bool = False,
        tools_config: ToolsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, ToolsConfig, WebToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self._guest_workspace = guest_workspace
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.web_config = web_config or WebToolsConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []
        self._scope_context_provider = scope_context_provider or ""
        self._scope_context_fn: Callable[[str], str] | None = None
        self._disable_memory_writes = bool(disable_memory_writes)

        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
        )
        self._unified_session = unified_session
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._guest_agent_cache: dict[Path, tuple[ContextBuilder, SessionManager]] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
            archive_disabled=self._disable_memory_writes,
        )
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self.dream = Dream(
            store=self.context.memory,
            provider=provider,
            model=self.model,
        )
        self._register_default_tools()
        if _tc.my.enable:
            self.tools.register(MyTool(loop=self, modify_allowed=_tc.my.allow_set))
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    # ── Audit logging ────────────────────────────────────────────────────────

    def _audit_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        *,
        tool_call_id: str | None = None,
        session_key: str | None = None,
    ) -> None:
        """Append a tool-call entry to the audit JSONL log (if auditing is enabled)."""
        if not getattr(self, "audit_config", None) or not self.audit_config.enabled:
            return
        import json as _json
        from datetime import datetime, timezone

        audit_path = Path(self.audit_config.path) if self.audit_config.path else self.workspace / "tool_audit.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "tool_call_id": tool_call_id,
            "session_key": session_key,
        }
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Failed to write audit log to {}", audit_path)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = (
            self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        )
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(NotebookEditTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allow_patterns=getattr(self.exec_config, 'allow_patterns', None) or None,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                )
            )
        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy)
            )
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stacks = await connect_mcp_servers(self._mcp_servers, self.tools)
            if self._mcp_stacks:
                self._mcp_connected = True
            else:
                logger.warning("No MCP servers connected successfully (will retry next message)")
        except asyncio.CancelledError:
            logger.warning("MCP connection cancelled (will retry next message)")
            self._mcp_stacks.clear()
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            self._mcp_stacks.clear()
        finally:
            self._mcp_connecting = False

    def _resolve_guest_agent_workspace(self, sender_id: str) -> tuple[ContextBuilder, SessionManager, frozenset[str]] | None:
        """Check if sender_id is a guest and return guest-specific context + sessions.

        Two routing paths:

        1. **Per-scope-type override** (preferred): if the agent's own
           workspace contains ``scope_workspaces.json`` mapping sender →
           subdir, return a tuple pointing to that subdir's
           ContextBuilder/SessionManager. Used when the agent is already
           running as the guest nanobot (workspace=guest_workspace) and we
           want to swap to a per-scope-type SOUL/AGENTS for the active
           sender.

        2. **ACL-based override** (legacy): reads ``guest_agent_acl.json``
           from the agent's workspace. If the sender matches, returns a
           tuple pointing to the ``guest_agent/`` subdirectory of the main
           workspace. Used when one nanobot serves both main and guest.

        Per-scope-type also applies when the ACL path matches — so a single
        nanobot setup with both ACL and scope_workspaces.json gets the
        scope-type subdir within the guest_agent/ workspace.
        """
        # Path 1: scope_workspaces.json directly under self.workspace.
        # Honours the case where this AgentLoop IS the guest nanobot.
        sw = self._read_scope_workspaces(self.workspace)
        if sw is not None:
            subdir = self._lookup_scope_workspace(sw, sender_id)
            if subdir:
                candidate = self.workspace / subdir
                if candidate.exists():
                    logger.info(
                        "scope_workspaces: sender={} → workspace={}",
                        sender_id, candidate,
                    )
                    return self._cached_guest_workspace(candidate)
                logger.warning(
                    "scope_workspaces.json maps sender {} to {} but {} missing — falling back",
                    sender_id, subdir, candidate,
                )

        # Path 2: legacy ACL routing — guest_agent_acl.json in main workspace.
        acl_path = self.workspace / "guest_agent_acl.json"
        if not acl_path.exists():
            return None
        try:
            acl = json.loads(acl_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        # Match sender_id against JIDs in the ACL
        # sender_id may be just the phone number (without @s.whatsapp.net)
        matched = False
        for jid in acl:
            jid_prefix = jid.split("@")[0] if "@" in jid else jid
            if sender_id == jid or sender_id == jid_prefix:
                matched = True
                break

        if not matched:
            return None

        default_workspace = self._guest_workspace or self.workspace / "guest_agent"
        if not default_workspace.exists():
            logger.warning("Guest agent ACL matched sender {} but guest_agent workspace missing", sender_id)
            return None

        # Within the legacy guest_agent/ workspace, scope_workspaces.json may
        # also exist — check it again so single-nanobot setups get the same
        # per-scope-type subdir routing.
        guest_agent_workspace = default_workspace
        sw2 = self._read_scope_workspaces(default_workspace)
        if sw2 is not None:
            subdir = self._lookup_scope_workspace(sw2, sender_id)
            if subdir:
                candidate = default_workspace / subdir
                if candidate.exists():
                    guest_agent_workspace = candidate
                    logger.info(
                        "scope_workspaces (within guest_agent): sender={} → workspace={}",
                        sender_id, candidate,
                    )
                else:
                    logger.warning(
                        "scope_workspaces.json maps sender {} to {} but {} missing — falling back",
                        sender_id, subdir, candidate,
                    )

        return self._cached_guest_workspace(guest_agent_workspace)

    @staticmethod
    def _read_scope_workspaces(workspace: Path) -> dict[str, str] | None:
        """Read scope_workspaces.json, returning None if absent or unparseable."""
        path = workspace / "scope_workspaces.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _lookup_scope_workspace(sw: dict[str, str], sender_id: str) -> str | None:
        """Find a sender → subdir mapping, trying common JID/LID/Telegram variants."""
        if sender_id in sw:
            return sw[sender_id]
        for variant in (
            f"{sender_id}@s.whatsapp.net",
            f"{sender_id}@lid",
            f"tg:{sender_id}",
        ):
            if variant in sw:
                return sw[variant]
        return None

    def _cached_guest_workspace(
        self, workspace: Path,
    ) -> tuple[ContextBuilder, SessionManager, frozenset[str]]:
        """Get or build cached (ContextBuilder, SessionManager, blocked) for a workspace."""
        if workspace not in self._guest_agent_cache:
            blocked = self._load_blocked_tools(workspace)
            self._guest_agent_cache[workspace] = (
                ContextBuilder(workspace),
                SessionManager(workspace),
                blocked,
            )
        return self._guest_agent_cache[workspace]

    @staticmethod
    def _load_blocked_tools(workspace: Path) -> frozenset[str]:
        """Load blocked tool names from blocked_tools.json in the workspace, if present."""
        path = workspace / "blocked_tools.json"
        if not path.exists():
            return frozenset()
        try:
            return frozenset(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return frozenset()

    def _get_scope_context(self, sender_id: str) -> str | None:
        """Resolve per-sender scope context via the configured provider.

        Provider config is ``"module:function"`` (e.g. ``"scope_store:render_scope_context_for_sender"``).
        The function MUST accept a single positional ``sender_id: str`` and
        return ``str``. An empty return means "no scope data for this sender"
        and no injection happens. Any exception is logged and skipped — the
        agent continues without injection (the workspace's static context
        still loads via the normal path).

        Misconfiguration (malformed format, missing module, missing attribute)
        auto-disables the provider for this process to avoid per-turn log
        spam; a restart is required to re-enable after fixing config.
        Transient provider exceptions do NOT auto-disable (retry next turn).

        Returns None when provider is unset, errors, or yields no content.
        """
        if not self._scope_context_provider or not sender_id:
            return None
        if self._scope_context_fn is None:
            try:
                mod_name, fn_name = self._scope_context_provider.split(":", 1)
            except ValueError:
                logger.error(
                    "scope_context_provider {} is not in 'module:function' form",
                    self._scope_context_provider,
                )
                self._scope_context_provider = ""  # disable after bad config to avoid log spam
                return None
            try:
                import importlib
                module = importlib.import_module(mod_name)
                self._scope_context_fn = getattr(module, fn_name)
            except (ImportError, AttributeError) as exc:
                logger.error(
                    "scope_context_provider {} not importable: {}",
                    self._scope_context_provider, exc,
                )
                self._scope_context_provider = ""
                return None
        try:
            result = self._scope_context_fn(sender_id)
        except Exception as exc:  # provider failures must not crash the agent
            logger.exception(
                "scope_context_provider raised for sender={}: {}", sender_id, exc,
            )
            return None
        if not isinstance(result, str) or not result.strip():
            return None
        return result

    @contextmanager
    def _restrict_fs_tools(self, guest_workspace: Path):
        """Temporarily restrict filesystem tools to a guest workspace."""
        from nanobot.agent.tools.filesystem import _FsTool
        from nanobot.agent.tools.shell import ExecTool

        saved: list[tuple[_FsTool, Path | None, Path | None]] = []
        saved_exec: list[tuple[ExecTool, str, bool]] = []

        for tool in self.tools._tools.values():
            if isinstance(tool, _FsTool):
                saved.append((tool, tool._workspace, tool._allowed_dir))
                tool._workspace = guest_workspace
                tool._allowed_dir = guest_workspace
            elif isinstance(tool, ExecTool):
                saved_exec.append((tool, tool.working_dir, tool.restrict_to_workspace))
                tool.working_dir = str(guest_workspace)
                tool.restrict_to_workspace = True
        try:
            yield
        finally:
            for tool, ws, ad in saved:
                tool._workspace = ws
                tool._allowed_dir = ad
            for tool, wd, rw in saved_exec:
                tool.working_dir = wd
                tool.restrict_to_workspace = rw

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        sender_id: str | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        # Compute the effective session key (accounts for unified sessions)
        # so that subagent results route to the correct pending queue.
        effective_key = UNIFIED_SESSION_KEY if self._unified_session else f"{channel}:{chat_id}"
        for name in ("message", "spawn", "cron", "my"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "spawn":
                        tool.set_context(channel, chat_id, effective_key=effective_key)
                    else:
                        tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))
        # Stamp sender identity onto the exec tool so trusted scripts can
        # authenticate the requester from the runtime, not from LLM args.
        if exec_tool := self.tools.get("exec"):
            if hasattr(exec_tool, "set_context"):
                exec_tool.set_context(channel=channel, sender_id=sender_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from nanobot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    async def _dispatch_command_inline(
        self,
        msg: InboundMessage,
        key: str,
        raw: str,
        dispatch_fn: Callable[[CommandContext], Awaitable[OutboundMessage | None]],
    ) -> None:
        """Dispatch a command directly from the run() loop and publish the result."""
        ctx = CommandContext(msg=msg, session=None, key=key, raw=raw, loop=self)
        result = await dispatch_fn(ctx)
        if result:
            await self.bus.publish_outbound(result)
        else:
            logger.warning("Command '{}' matched but dispatch returned None", raw)

    async def _cancel_active_tasks(self, key: str) -> int:
        """Cancel and await all active tasks and subagents for *key*.

        Returns the total number of cancelled tasks + subagents.
        """
        tasks = self._active_tasks.pop(key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(key)
        return cancelled + sub_cancelled

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """Return the session key used for task routing and mid-turn injections."""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        sender_id: str | None = None,
        blocked_tools: frozenset[str] = frozenset(),
        model_override: str | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections).
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            sender_id=sender_id,
        )
        hook: AgentHook = (
            CompositeHook([loop_hook] + self._extra_hooks) if self._extra_hooks else loop_hook
        )

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """Drain follow-up messages from the pending queue.

            When no messages are immediately available but sub-agents
            spawned in this dispatch are still running, blocks until at
            least one result arrives (or timeout).  This keeps the runner
            loop alive so subsequent sub-agent completions are consumed
            in-order rather than dispatched separately.
            """
            if pending_queue is None:
                return []

            def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = extract_documents(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                runtime_ctx = self.context._build_runtime_context(
                    pending_msg.channel,
                    pending_msg.chat_id,
                    self.context.timezone,
                )
                if isinstance(user_content, str):
                    merged: str | list[dict[str, Any]] = f"{runtime_ctx}\n\n{user_content}"
                else:
                    merged = [{"type": "text", "text": runtime_ctx}] + user_content
                return {"role": "user", "content": merged}

            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    items.append(_to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break

            # Block if nothing drained but sub-agents spawned in this dispatch
            # are still running.  Keeps the runner loop alive so subsequent
            # completions are injected in-order rather than dispatched separately.
            if (not items
                    and session is not None
                    and self.subagents.get_running_count_by_session(session.key) > 0):
                try:
                    msg = await asyncio.wait_for(pending_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for sub-agent completion in session {}",
                        session.key,
                    )
                    return items
                items.append(_to_user_message(msg))
                while len(items) < limit:
                    try:
                        items.append(_to_user_message(pending_queue.get_nowait()))
                    except asyncio.QueueEmpty:
                        break

            return items

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=model_override or self.model,
            max_iterations=self.max_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            blocked_tools=blocked_tools,
            workspace=self.workspace,
            session_key=session.key if session else None,
            context_window_tokens=self.context_window_tokens,
            context_block_limit=self.context_block_limit,
            provider_retry_mode=self.provider_retry_mode,
            progress_callback=on_progress,
            retry_wait_callback=on_retry_wait,
            checkpoint_callback=_checkpoint,
            injection_callback=_drain_pending,
        ))
        self._last_usage = result.usage
        if result.stop_reason == STOP_MAX_ITERATIONS:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if on_stream and on_stream_end:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == STOP_ERROR:
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                self.auto_compact.check_expired(
                    self._schedule_background,
                    active_session_keys=self._pending_queues.keys(),
                )
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                await self._dispatch_command_inline(
                    msg, msg.session_key, raw,
                    self.commands.dispatch_priority,
                )
                continue
            effective_key = self._effective_session_key(msg)
            # If this session already has an active pending queue (i.e. a task
            # is processing this session), route the message there for mid-turn
            # injection instead of creating a competing task.
            if effective_key in self._pending_queues:
                # Non-priority commands must not be queued for injection;
                # dispatch them directly (same pattern as priority commands).
                if self.commands.is_dispatchable_command(raw):
                    await self._dispatch_command_inline(
                        msg, effective_key, raw,
                        self.commands.dispatch,
                    )
                    continue
                pending_msg = msg
                if effective_key != msg.session_key:
                    pending_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                try:
                    self._pending_queues[effective_key].put_nowait(pending_msg)
                except asyncio.QueueFull:
                    logger.warning(
                        "Pending queue full for session {}, falling back to queued task",
                        effective_key,
                    )
                else:
                    logger.info(
                        "Routed follow-up message to pending queue for session {}",
                        effective_key,
                    )
                    continue
            # Compute the effective session key before dispatching
            # This ensures /stop command can find tasks correctly when unified session is enabled
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(effective_key, []).append(task)
            task.add_done_callback(
                lambda t, k=effective_key: self._active_tasks.get(k, [])
                and self._active_tasks[k].remove(t)
                if t in self._active_tasks.get(k, [])
                else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        session_key = self._effective_session_key(msg)
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()

        # Register a pending queue so follow-up messages for this session are
        # routed here (mid-turn injection) instead of spawning a new task.
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending

        try:
            async with lock, gate:
                try:
                    on_stream = on_stream_end = None
                    if msg.metadata.get("_wants_stream"):
                        # Split one answer into distinct stream segments.
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            ))

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            nonlocal stream_segment
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True
                            meta["_resuming"] = resuming
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            ))
                            stream_segment += 1

                    response = await self._process_message(
                        msg, on_stream=on_stream, on_stream_end=on_stream_end,
                        pending_queue=pending,
                    )
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", session_key)
                    # Preserve partial context from the interrupted turn so
                    # the user does not lose tool results and assistant
                    # messages accumulated before /stop.  The checkpoint was
                    # already persisted to session metadata by
                    # _emit_checkpoint during tool execution; materializing
                    # it into session history now makes it visible in the
                    # next conversation turn.
                    try:
                        key = self._effective_session_key(msg)
                        session = self.sessions.get_or_create(key)
                        if self._restore_runtime_checkpoint(session):
                            self._clear_pending_user_turn(session)
                            self.sessions.save(session)
                            logger.info(
                                "Restored partial context for cancelled session {}",
                                key,
                            )
                    except Exception:
                        logger.debug(
                            "Could not restore checkpoint for cancelled session {}",
                            session_key,
                            exc_info=True,
                        )
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", session_key)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    ))
        finally:
            # Drain any messages still in the pending queue and re-publish
            # them to the bus so they are processed as fresh inbound messages
            # rather than silently lost.
            queue = self._pending_queues.pop(session_key, None)
            if queue is not None:
                leftover = 0
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self.bus.publish_inbound(item)
                    leftover += 1
                if leftover:
                    logger.info(
                        "Re-published {} leftover message(s) to bus for session {}",
                        leftover, session_key,
                    )

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        model_override: str | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.sessions.save(session)
            if self._restore_pending_user_turn(session):
                self.sessions.save(session)

            session, pending = self.auto_compact.prepare_session(session, key)

            await self.consolidator.maybe_consolidate_by_tokens(
                session,
                session_summary=pending,
            )
            # Persist subagent follow-ups into durable history BEFORE prompt
            # assembly. ContextBuilder merges adjacent same-role messages for
            # provider compatibility, which previously caused the follow-up to
            # disappear from session.messages while still being visible to the
            # LLM via the merged prompt. See _persist_subagent_followup.
            is_subagent = msg.sender_id == "subagent"
            if is_subagent and self._persist_subagent_followup(session, msg):
                self.sessions.save(session)
            self._set_tool_context(
                channel, chat_id, msg.metadata.get("message_id"), sender_id=msg.sender_id,
            )
            history = session.get_history(max_messages=0)
            current_role = "assistant" if is_subagent else "user"

            # Subagent content is already in `history` above; passing it again
            # as current_message would double-project it into the prompt.
            messages = self.context.build_messages(
                history=history,
                current_message="" if is_subagent else msg.content,
                channel=channel,
                chat_id=chat_id,
                session_summary=pending,
                current_role=current_role,
            )
            final_content, _, all_msgs, _, _ = await self._run_agent_loop(
                messages, session=session, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
                sender_id=msg.sender_id,
                model_override=model_override,
                pending_queue=pending_queue,
            )
            # With current_role="assistant" for subagent, ContextBuilder merges
            # the Current-Time prefix into the last history assistant message
            # rather than injecting a separate user message — so skip is just
            # system + history.
            self._save_turn(session, all_msgs, 1 + len(history), usage=self._last_usage)
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        # Extract document text from media at the processing boundary so all
        # channels benefit without format-specific logic in ContextBuilder.
        if msg.media:
            new_content, image_only = extract_documents(msg.content, msg.media)
            msg = dataclasses.replace(msg, content=new_content, media=image_only)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # PostHog analytics — capture inbound context. Skip for synthetic
        # messages (heartbeat, cron) so we don't record agent-initiated
        # activity as user-sent messages.
        from nanobot.analytics.hook import get_analytics_hook
        _analytics = get_analytics_hook()
        _synthetic = bool(msg.metadata.get("synthetic"))
        _synthetic_trigger_kind = str(msg.metadata.get("trigger_kind") or "synthetic")
        _synthetic_start = time.monotonic() if _synthetic else 0.0

        # Pre-turn quota gate (default-tier weekly token budget). Bails fast
        # for synthetic / byok / managed turns; on a hard cap-hit we send
        # the upgrade reply and return WITHOUT entering the agent loop or
        # firing agent_responded — the gate is not the agent.
        from nanobot.analytics.quota_gate import check_token_budget_before_turn
        _turn_ctx: dict[str, Any] = {"is_synthetic": _synthetic}
        try:
            _cap_hit_reply = check_token_budget_before_turn(_turn_ctx)
        except Exception:
            logger.debug("quota_gate hook error (non-fatal)", exc_info=True)
            _cap_hit_reply = None
        if _cap_hit_reply is not None:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=_cap_hit_reply,
                metadata=dict(msg.metadata or {}),
            )

        # Resolve guest_agent workspace if sender is in guest agent ACL
        guest = self._resolve_guest_agent_workspace(msg.sender_id) if msg.sender_id else None
        context = guest[0] if guest else self.context
        sessions = guest[1] if guest else self.sessions
        blocked_tools = guest[2] if guest else frozenset()

        _analytics_ctx = None if _synthetic else _analytics.on_message_received(
            channel=msg.channel,
            sender_id=msg.sender_id,
            content=msg.content,
            media=msg.media,
            timestamp=msg.timestamp,
            is_guest=guest is not None,
        )

        # hist_chat_messages persistence (homer family-history). No-op unless
        # HOMER_CHAT_PERSIST_ENABLED + Supabase env are set; failures swallow.
        # Synthetic turns (heartbeat/cron) are not chat and shouldn't land in
        # the contributor's transcript.
        from nanobot.analytics.chat_persist import get_chat_persist_hook
        _chat_persist = get_chat_persist_hook()
        _chat_ctx = None
        if not _synthetic:
            try:
                _chat_ctx = await _chat_persist.on_message_received(
                    channel=msg.channel,
                    sender_id=msg.sender_id,
                    content=msg.content,
                    media=msg.media,
                    timestamp=msg.timestamp,
                    schedule_background=self._schedule_background,
                )
            except Exception:
                logger.debug("chat_persist on_message_received error (non-fatal)", exc_info=True)

        key = session_key or msg.session_key
        session = sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            sessions.save(session)
        if self._restore_pending_user_turn(session):
            sessions.save(session)

        session, pending = self.auto_compact.prepare_session(session, key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        await self.consolidator.maybe_consolidate_by_tokens(
            session,
            session_summary=pending,
        )

        self._set_tool_context(
            msg.channel, msg.chat_id, msg.metadata.get("message_id"), sender_id=msg.sender_id,
        )
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)

        initial_messages = context.build_messages(
            history=history,
            current_message=msg.content,
            session_summary=pending,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        # Ephemeral: inserted into this turn's messages only — never written
        # to session history, so stale scope data can't accumulate across turns.
        #
        # Gate is just on ``msg.sender_id`` — the previous additional ``guest``
        # gate (where ``guest = _resolve_guest_agent_workspace(...)``) tied
        # injection to the existence of a per-scope-type workspace override,
        # which is a different concern. Many guests are members of relationship
        # / event scopes that don't have a per-scope-type subdir, and those
        # were silently missing scope context. The provider itself returns
        # "" for senders with no scope data, so this gate is sufficient.
        scope_ctx_injected = False
        if msg.sender_id:
            scope_ctx = self._get_scope_context(msg.sender_id)
            if scope_ctx:
                initial_messages.insert(
                    1, {"role": "system", "content": scope_ctx}
                )
                scope_ctx_injected = True

        async def _bus_progress(
            content: str,
            *,
            tool_hint: bool = False,
            tool_events: list[dict[str, Any]] | None = None,
        ) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            if tool_events:
                meta["_tool_events"] = tool_events
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        async def _on_retry_wait(content: str) -> None:
            meta = dict(msg.metadata or {})
            meta["_retry_wait"] = True
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Persist the triggering user message up front so a mid-turn crash
        # doesn't silently lose the prompt on recovery. ``media`` rides along
        # as raw on-disk paths — sanitized image blocks are stripped from
        # JSONL, and webui replay needs the paths to mint signed URLs.
        user_persisted_early = False
        media_paths = [p for p in (msg.media or []) if isinstance(p, str) and p]
        has_text = isinstance(msg.content, str) and msg.content.strip()
        if has_text or media_paths:
            extra: dict[str, Any] = {"media": list(media_paths)} if media_paths else {}
            text = msg.content if isinstance(msg.content, str) else ""
            session.add_message("user", text, **extra)
            self._mark_pending_user_turn(session)
            sessions.save(session)
            user_persisted_early = True

        # Restrict file tools to guest workspace for the duration of the agent loop
        fs_ctx = self._restrict_fs_tools(context.workspace) if guest else nullcontext()
        with fs_ctx:
            final_content, tools_used, all_msgs, stop_reason, had_injections = await self._run_agent_loop(
                initial_messages,
                on_progress=on_progress or _bus_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                on_retry_wait=_on_retry_wait,
                session=session,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_id=msg.metadata.get("message_id"),
                sender_id=msg.sender_id,
                blocked_tools=blocked_tools,
                model_override=model_override,
                pending_queue=pending_queue,
            )

        if is_blank_text(final_content):
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE
            stop_reason = STOP_EMPTY_FINAL

        # Skip system prompt + ephemeral scope_ctx (if any) + prior history
        # + the already-persisted user message; what's left is the current
        # turn's new assistant/tool messages.
        save_skip = (
            1
            + (1 if scope_ctx_injected else 0)
            + len(history)
            + (1 if user_persisted_early else 0)
        )
        self._save_turn(session, all_msgs, save_skip, usage=self._last_usage)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        sessions.save(session)
        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))

        # When follow-up messages were injected mid-turn, a later natural
        # language reply may address those follow-ups and should not be
        # suppressed just because MessageTool was used earlier in the turn.
        # However, if the turn falls back to the empty-final-response
        # placeholder, suppress it when the real user-visible output already
        # came from MessageTool — the user-visible output was intentional,
        # so the trailing empty turn is silence by design, not a failure.
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == STOP_EMPTY_FINAL:
                stop_reason = STOP_INTENTIONAL_SILENCE

        # Keep non-delivered turns at debug so log review isn't misled into
        # thinking a real message went out.
        if stop_reason == STOP_INTENTIONAL_SILENCE:
            logger.debug(
                "Intentional silence for {}:{} — MessageTool already delivered output",
                msg.channel, msg.sender_id,
            )
            return None
        elif stop_reason == STOP_EMPTY_FINAL:
            logger.debug(
                "Empty final response for {}:{} — downstream filter will decide delivery",
                msg.channel, msg.sender_id,
            )
        else:
            preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
            logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        # PostHog analytics — fire events after response is built. The
        # user-facing path emits message_sent / agent_responded; the
        # synthetic path emits a single agent_initiated_action so
        # proactive work (briefings, reminder fires, scheduled tasks)
        # is visible without polluting the user-message funnel.
        if _analytics_ctx is not None:
            _tools = tools_used or []
            escalation_used = "escalate" in _tools or "resolve_escalation" in _tools
            try:
                await _analytics.on_response_sent(
                    _analytics_ctx,
                    response_content=final_content,
                    tools_used=_tools,
                    escalation_triggered=escalation_used,
                    schedule_background=self._schedule_background,
                )
            except Exception:
                logger.debug("Analytics hook error (non-fatal)", exc_info=True)
        elif _synthetic:
            # The empty-final placeholder isn't a "real" response — drop it
            # before reporting so had_outbound stays accurate.
            _final = (
                None
                if stop_reason in (STOP_EMPTY_FINAL, STOP_INTENTIONAL_SILENCE)
                else final_content
            )
            try:
                _analytics.track_agent_initiated_action(
                    trigger_kind=_synthetic_trigger_kind,
                    response_content=_final,
                    tools_used=tools_used or [],
                    latency_ms=int((time.monotonic() - _synthetic_start) * 1000),
                )
            except Exception:
                logger.debug("agent_initiated_action emit failed", exc_info=True)

        # Persist assistant reply to hist_chat_messages. _chat_ctx is None
        # when persistence is disabled, the channel is unsupported, or the
        # sender didn't resolve to a contributor — all already logged in
        # on_message_received.
        if _chat_ctx is not None:
            try:
                await _chat_persist.on_response_sent(
                    _chat_ctx,
                    response_content=final_content,
                    schedule_background=self._schedule_background,
                )
            except Exception:
                logger.debug("chat_persist on_response_sent error (non-fatal)", exc_info=True)

        meta = dict(msg.metadata or {})
        streamed = on_stream is not None and stop_reason != STOP_ERROR
        if streamed:
            meta["_streamed"] = True

        # Post-turn quota warn — nudge the user when the household crossed the
        # warn threshold this turn.
        #
        # On non-streamed channels we append the warn copy to ``final_content``
        # so the user reads one coherent message. On streamed channels (Telegram
        # et al.) the user already saw ``final_content`` live; the dispatcher
        # then SKIPS the returned OutboundMessage when ``_streamed=True`` (see
        # ``ChannelManager._send_once``), which means an in-content appendix
        # would never reach the user (issue #58). Publish the appendix as a
        # separate, non-streamed follow-up message instead — same delivery path
        # the heartbeat / cron uses for proactive nudges. ``final_content``
        # itself stays unchanged so session history doesn't double-record the
        # appendix that already lives in the streamed transcript.
        try:
            from nanobot.analytics.quota_gate import maybe_append_quota_warn
            with_warn = maybe_append_quota_warn(_turn_ctx, final_content)
            if with_warn is not None and with_warn != final_content:
                if streamed:
                    appendix_text = with_warn[len(final_content):].lstrip()
                    if appendix_text:
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content=appendix_text,
                                metadata=dict(msg.metadata or {}),
                            )
                        )
                else:
                    final_content = with_warn
        except Exception:
            logger.debug("quota_gate warn appendix error (non-fatal)", exc_info=True)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
            stop_reason=stop_reason,
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int,
                    usage: dict[str, int] | None = None) -> None:
        """Save new-turn messages into session, truncating large tool results.

        If *usage* is provided it is attached to the **last** assistant message
        so downstream analytics can read actual token counts from the session log.
        """
        from datetime import datetime
        last_assistant_entry: dict | None = None
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the entire runtime-context block (including any session summary).
                    # The block is bounded by _RUNTIME_CONTEXT_TAG and _RUNTIME_CONTEXT_END.
                    end_marker = ContextBuilder._RUNTIME_CONTEXT_END
                    end_pos = content.find(end_marker)
                    if end_pos >= 0:
                        after = content[end_pos + len(end_marker):].lstrip("\n")
                        if after:
                            entry["content"] = after
                        else:
                            continue
                    else:
                        # Fallback: no end marker found, strip the tag prefix
                        after_tag = content[len(ContextBuilder._RUNTIME_CONTEXT_TAG):].lstrip("\n")
                        if after_tag.strip():
                            entry["content"] = after_tag
                        else:
                            continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            if role == "assistant":
                last_assistant_entry = entry
            session.messages.append(entry)
        # Attach usage to the final assistant message of this turn
        if usage and last_assistant_entry is not None:
            last_assistant_entry["usage"] = usage
        session.updated_at = datetime.now()

    def _persist_subagent_followup(self, session: Session, msg: InboundMessage) -> bool:
        """Persist subagent follow-ups before prompt assembly so history stays durable.

        Returns True if a new entry was appended; False if the follow-up was
        deduped (same ``subagent_task_id`` already in session) or carries no
        content worth persisting.
        """
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id") if isinstance(msg.metadata, dict) else None
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
        )
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """Close a turn that only persisted the user message before crashing."""
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        model_override: str | None = None,
        is_synthetic: bool = False,
        trigger_kind: str = "synthetic",
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload.

        *is_synthetic* flags messages that were not initiated by a user —
        heartbeat ticks, cron reminders, internal self-sends. These bypass
        the user-facing analytics hooks (``message_sent`` /
        ``agent_responded``) and instead emit a single
        ``agent_initiated_action`` event tagged with *trigger_kind*
        (``"heartbeat"``, ``"cron"``, ``"scheduled"``, …) so dashboards
        can answer "what is Homer doing on its own?" without polluting
        the user-message funnel.
        """
        await self._connect_mcp()
        synthetic_meta: dict[str, Any] = {}
        if is_synthetic:
            synthetic_meta = {"synthetic": True, "trigger_kind": trigger_kind}
        msg = InboundMessage(
            channel=channel,
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata=synthetic_meta,
        )
        return await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            model_override=model_override,
        )
