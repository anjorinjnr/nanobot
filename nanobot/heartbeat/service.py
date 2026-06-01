"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
import calendar
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import AbstractContextManager, nullcontext
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, Protocol

from loguru import logger
from zoneinfo import ZoneInfo

from nanobot.agent.runner import STOP_EMPTY_FINAL, STOP_ERROR, STOP_INTENTIONAL_SILENCE
from nanobot.bus.events import OutboundMessage
from nanobot.utils.heartbeat_lock import heartbeat_lock
from nanobot.utils.helpers import write_text_atomic
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_SCHED_PAT = re.compile(r"Schedule:\s*(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)")
_RECUR_PAT = re.compile(
    r"Recur:\s*every\s+(\d+)\s+(minute|hour|day|week|month)s?", re.IGNORECASE,
)
_UNTIL_PAT = re.compile(r"Until:\s*(\d{4}-\d{2}-\d{2})")
_LASTRUN_PAT = re.compile(r"Last-run:[^\n]*")
_LASTRUN_VALUE_PAT = re.compile(
    r"^Last-run:\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)", re.MULTILINE
)
_RECIPIENTS_PAT = re.compile(r"^Recipients:\s*(.+)", re.MULTILINE)
# Stable task IDs emitted by homer's tasks_update.py: literal `t_` + 8 lowercase
# base32 chars. Anchored on its own line so we don't accidentally match
# `Id:` substrings inside free-form fields. Blocks predating the rollout
# may not have an Id line — callers must handle None gracefully.
_ID_PAT = re.compile(r"^Id:\s*(t_[a-z2-7]{8})\s*$", re.MULTILINE)


def _apply_recur(dt: datetime, n: int, unit: str) -> datetime | None:
    """Return ``dt`` advanced by ``n`` units, or ``None`` for unknown units.

    Month arithmetic uses calendar months (not fixed 30-day deltas) so
    ``every 1 month`` lands on the same day-of-month where possible,
    clamped to the last day when the target month is shorter (Jan 31 +
    1 month → Feb 28/29).
    """
    if unit == "minute":
        return dt + timedelta(minutes=n)
    if unit == "hour":
        return dt + timedelta(hours=n)
    if unit == "day":
        return dt + timedelta(days=n)
    if unit == "week":
        return dt + timedelta(weeks=n)
    if unit == "month":
        total = dt.month - 1 + n
        new_year = dt.year + total // 12
        new_month = total % 12 + 1
        last_day = calendar.monthrange(new_year, new_month)[1]
        return dt.replace(year=new_year, month=new_month, day=min(dt.day, last_day))
    return None


def _effective_due(
    block: str, schedule_dt: datetime, schedule_str: str
) -> tuple[datetime, str]:
    """Resolve when a recurring task is next allowed to fire.

    Schedule is the floor: Last-run + Recur cannot push the due time
    earlier than it (that's the regression — a future Schedule from a
    --tick or pause must not be undermined by stale Last-run math).
    Returns the effective datetime and a display string for prompts.
    """
    lr_match = _LASTRUN_VALUE_PAT.search(block)
    recur_match = _RECUR_PAT.search(block)
    if not (lr_match and recur_match):
        return schedule_dt, schedule_str
    try:
        lr_str = lr_match.group(1).strip()
        last_run_dt = datetime.strptime(
            lr_str, "%Y-%m-%d %H:%M" if " " in lr_str else "%Y-%m-%d"
        )
        amount = int(recur_match.group(1))
        unit = recur_match.group(2).lower()
        next_run = _apply_recur(last_run_dt, amount, unit)
        if next_run is None:
            return schedule_dt, schedule_str
    except ValueError:
        return schedule_dt, schedule_str
    effective = max(schedule_dt, next_run)
    return effective, effective.strftime("%Y-%m-%d %H:%M")

def filter_heartbeat_response(
    resp: OutboundMessage | None,
    tasks: str,
    suppress_errors: bool = False,
) -> str:
    """Extract deliverable content from a heartbeat execution result.

    Returns the response content for normal completions, a short
    diagnostic for API errors (unless suppress_errors is set), or
    empty string when the response should be silenced.
    """
    if not resp:
        return ""

    if resp.stop_reason == STOP_INTENTIONAL_SILENCE:
        # Heartbeat turn completed silently by design — the MessageTool
        # already delivered the real user-visible output; no failure here.
        logger.debug("Heartbeat: silent turn (by design) for: {}", tasks[:120])
        return ""

    if resp.stop_reason == STOP_EMPTY_FINAL or resp.content == EMPTY_FINAL_RESPONSE_MESSAGE:
        logger.info("Heartbeat: suppressed empty response for: {}", tasks[:120])
        return ""

    if resp.stop_reason == STOP_ERROR:
        logger.warning("Heartbeat: task error for: {}", tasks[:120])
        if suppress_errors:
            return ""
        return f"⚠️ Heartbeat error running: {tasks}. Check logs for details."

    return resp.content or ""


_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run"],
                        "description": "skip = nothing to do, run = has active tasks",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Natural-language summary of active tasks (required for run)",
                    },
                },
                "required": ["action"],
            },
        },
    }
]


def _resolve_dispatch_target(
    recipient: str | None, channel: str | None,
) -> tuple[str, str] | None:
    """Resolve ``(recipient_symbol, channel)`` → ``(channel, chat_id)`` via
    homer's users_loader. Returns ``None`` for: empty inputs, missing
    users_loader (standalone nanobot), or resolution failure (unknown
    symbol / channel / empty handle).

    Callers distinguish the empty-inputs case from the failure cases via
    the inputs themselves, not the return — both produce ``None``.
    """
    if not recipient or not channel:
        return None
    try:
        import users_loader  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        handle = users_loader.resolve_handle(recipient, channel)
    except Exception as e:
        logger.warning(
            "Heartbeat: cannot resolve recipient {!r}:{!r} ({}); skipping",
            recipient, channel, e,
        )
        return None
    return (channel, handle)


@dataclass
class DueTask:
    name: str
    task_type: Literal["announcement", "system", "reminder"]
    schedule: str | None  # None for announcements
    model: str | None = None  # Optional per-task model override
    pre_check: str | None = None  # Optional command to run before LLM dispatch
    recipients: str | None = None  # Raw Recipients line, e.g. "primary:whatsapp,seun:whatsapp"
    # Stable identifier from HEARTBEAT.md `Id: t_xxxxxxxx`. None for legacy
    # blocks that predate the ID rollout — fall back to name-based matching.
    id: str | None = None
    # Optional path (relative to the workspace) of a file whose contents
    # become the agent message when this task fires, replacing the default
    # task-name summary. Supports `{recipient}` substitution so a single
    # task with multiple Recipients fans out to per-user prompt files
    # (e.g. `context/users/{recipient}.brief.md`). When set, the task
    # dispatches once per recipient instead of once per group.
    prompt_file: str | None = None

    def _iter_recipient_entries(self) -> Iterator[tuple[str, str]]:
        """Yield ``(name, channel)`` pairs parsed from ``Recipients:``.

        Each comma-separated entry is split on its LAST colon (so an id
        can contain colons — email-like ids). Empty or malformed entries
        are skipped; an entry without a colon logs a warning so a typo
        doesn't silently disable a recipient. Channel is lowercased; name
        keeps its original case (display_name lookup is case-insensitive
        downstream).
        """
        if not self.recipients:
            return
        for entry in self.recipients.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                logger.warning(
                    "DueTask {!r}: skipping malformed recipient {!r} (no channel suffix)",
                    self.name, entry,
                )
                continue
            name, channel = entry.rsplit(":", 1)
            name = name.strip()
            channel = channel.strip().lower()
            if name and channel:
                yield name, channel

    def recipient_channels(self) -> set[str]:
        """Channels addressed in ``Recipients:`` (e.g. ``{"whatsapp"}``).
        Empty set means no Recipients field (fall through to caller defaults)."""
        return {channel for _name, channel in self._iter_recipient_entries()}

    def recipient_names(self) -> list[str]:
        """Recipient names, deduplicated, in source order.

        A name addressed on multiple channels surfaces once — prompt-file
        dispatch fires per name, not per channel.
        """
        out: list[str] = []
        for name, _channel in self._iter_recipient_entries():
            if name not in out:
                out.append(name)
        return out

    def recipient_pairs(self) -> list[tuple[str, str]]:
        """``(name, channel)`` pairs in source order, NOT deduplicated.

        The dispatcher uses this to fan out one resolution per
        ``(name, channel)`` entry — a name addressed on two channels
        gets two dispatches, each routing to its own session.
        """
        return list(self._iter_recipient_entries())


# Canonical preset slate — kept in lockstep with homer's tools/switch_model.py
# MODELS dict. Every entry resolves to an OpenRouter slug (vendor/model) because
# the consolidation migrated all tenant LLM traffic to OpenRouter sub-keys.
# Older aliases (flash25, flash, pro, sonnet, haiku) are retained for back-compat
# with tasks predating the consolidation but should be considered deprecated.
MODEL_PRESETS: dict[str, str] = {
    # OpenRouter-routed canonical presets
    "auto":            "openrouter/auto",
    "cheap":           "deepseek/deepseek-v3.2",
    "default-cheap":   "deepseek/deepseek-v3.2",

    "gemini-fast":     "google/gemini-2.5-flash",
    "gemini-balanced": "google/gemini-2.5-pro",
    "gemini-smart":    "google/gemini-3.1-pro-preview",

    "gpt-fast":        "openai/gpt-5-mini",
    "gpt-balanced":    "openai/gpt-5",
    "gpt-smart":       "openai/gpt-5.5",

    "claude-fast":     "anthropic/claude-haiku-4.5",
    "claude-balanced": "anthropic/claude-sonnet-4.6",
    "claude-smart":    "anthropic/claude-opus-4.7",

    # Deprecated direct-provider aliases — pre-consolidation.
    "flash25":         "gemini/gemini-2.5-flash",
    "flash":           "gemini/gemini-3-flash-preview",
    "pro":             "gemini/gemini-3.1-pro-preview",
    "sonnet":          "claude-sonnet-4-6",
    "haiku":           "claude-haiku-4-5-20251001",
}


class _OnExecuteContext(Protocol):
    """Structural type for the ``on_execute_context`` hook.

    The hook receives the current dispatch's task list and (when the
    dispatcher pre-resolved a single recipient for the call) the
    ``target`` to pin MessageTool against. Implementations return a
    context manager that wraps the ``on_execute`` invocation.

    Defined as a Protocol rather than a plain ``Callable[...]`` so the
    expected signature stays self-documenting and type-checkers can
    flag drift in either direction (caller adding a kwarg the hook
    doesn't accept, or hook narrowing the accepted shape).
    """

    def __call__(
        self,
        group_tasks: list["DueTask"],
        *,
        target: tuple[str, str] | None = None,
    ) -> AbstractContextManager[None]: ...


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.

    Phase 1 (decision): when last_run_tracking=True, deterministically computes
    which tasks are due without an LLM call. Falls back to LLM when disabled.

    Phase 2 (execution): only triggered when Phase 1 returns ``run``.  The
    ``on_execute`` callback runs the task through the full agent loop and
    returns the result to deliver.
    """

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str, str | None], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_execute_context: _OnExecuteContext | None = None,
        interval_s: int = 30 * 60,
        enabled: bool = True,
        last_run_tracking: bool = False,
        timezone: str | None = None,
        suppress_errors: bool = False,
        pre_check_registry: dict[str, str] | None = None,
        model_presets: dict[str, str] | None = None,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.on_execute_context = on_execute_context
        self.interval_s = interval_s
        self.enabled = enabled
        self.last_run_tracking = last_run_tracking
        self.timezone = timezone
        self.suppress_errors = suppress_errors
        self.pre_check_registry = pre_check_registry or {}
        # An empty/None override falls back to the built-in slate so existing
        # deployments and tests keep working without a config change.
        self.model_presets = model_presets or MODEL_PRESETS
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _now(self) -> datetime:
        """Return current time in the configured timezone."""
        try:
            tz = ZoneInfo(self.timezone) if self.timezone else None
        except (KeyError, Exception):
            tz = None
        return datetime.now(tz=tz) if tz else datetime.now().astimezone()

    @staticmethod
    def _compute_due_tasks(
        content: str,
        now: datetime,
        model_presets: dict[str, str] | None = None,
    ) -> list[DueTask]:
        """Deterministically compute which tasks are due now.

        Returns a list of DueTask covering three types:
        - announcement: any ### entry under ## Announcements (always due)
        - system: Type: system task whose Schedule has passed
        - reminder: user reminder task whose Schedule has passed
        """
        due: list[DueTask] = []

        # 1. Announcements — always due if any ### entries exist
        ann_match = re.search(r"^## Announcements\s*$", content, re.MULTILINE)
        if ann_match:
            ann_start = ann_match.end()
            next_sec = re.search(r"^## ", content[ann_start:], re.MULTILINE)
            ann_end = ann_start + next_sec.start() if next_sec else len(content)
            for m in re.finditer(r"^###\s+(.+)", content[ann_start:ann_end], re.MULTILINE):
                due.append(DueTask(name=m.group(1).strip(), task_type="announcement", schedule=None))

        # 2. User Tasks — due if now >= Schedule
        user_match = re.search(r"^## User Tasks\s*$", content, re.MULTILINE)
        if not user_match:
            return due

        section_start = user_match.end()
        next_sec = re.search(r"^## ", content[section_start:], re.MULTILINE)
        section_end = section_start + next_sec.start() if next_sec else len(content)
        section = content[section_start:section_end]

        for block in re.split(r"\n(?=###\s)", section):
            block = block.strip()
            if not block.startswith("###"):
                continue

            name_match = re.match(r"###\s+(.+)", block)
            if not name_match:
                continue
            task_name = name_match.group(1).strip()
            task_type: Literal["system", "reminder"] = (
                "system" if re.search(r"^Type:\s*system", block, re.MULTILINE) else "reminder"
            )

            schedule_match = re.search(
                r"Schedule:\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)", block
            )
            if not schedule_match:
                continue
            schedule_str = schedule_match.group(1).strip()

            try:
                if " " in schedule_str:
                    schedule_dt = datetime.strptime(schedule_str, "%Y-%m-%d %H:%M")
                else:
                    schedule_dt = datetime.strptime(schedule_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Skip tasks past their Until date
            until_match = re.search(r"Until:\s*(\d{4}-\d{2}-\d{2})", block)
            if until_match:
                try:
                    until_dt = datetime.strptime(until_match.group(1), "%Y-%m-%d")
                    if now > until_dt:
                        continue
                except ValueError:
                    pass

            model_match = re.search(r"^Model:\s*(.+)", block, re.MULTILINE)
            model = None
            if model_match:
                raw = model_match.group(1).strip()
                presets = model_presets if model_presets is not None else MODEL_PRESETS
                model = presets.get(raw, raw)  # resolve preset or use as-is

            pre_check_match = re.search(r"^Pre-check:\s*(\S+)", block, re.MULTILINE)
            pre_check = pre_check_match.group(1).strip() if pre_check_match else None

            recipients_match = _RECIPIENTS_PAT.search(block)
            recipients = recipients_match.group(1).strip() if recipients_match else None

            id_match = _ID_PAT.search(block)
            task_id = id_match.group(1) if id_match else None

            # Paths cannot contain whitespace — `\S+` truncates at the first
            # space. Workspace-relative paths like `users/{recipient}.brief.md`
            # are the intended shape; anything else is a malformed task block.
            prompt_file_match = re.search(r"^Prompt-file:\s*(\S+)", block, re.MULTILINE)
            prompt_file = prompt_file_match.group(1).strip() if prompt_file_match else None

            effective_due, _ = _effective_due(block, schedule_dt, schedule_str)
            if now < effective_due:
                continue

            # Rule 3 — every system/reminder task must declare Recipients.
            # A task without Recipients has no safe routing target: the
            # dispatcher would call on_execute() with no target= and the
            # agent would guess chat_id from memory. That's the failure
            # mode that leaked kid-related Gmail content to a guest on
            # 2026-05-27. Prompt-file does NOT substitute — it carries
            # per-recipient prompt content but still needs Recipients to
            # know who to fan out to. Refuse loudly so the operator
            # notices and adds Recipients, rather than silently dispatching
            # to whoever the LLM picks. Announcements are exempt (different
            # section, no Schedule, intentionally broadcast).
            if not recipients:
                logger.error(
                    "Heartbeat: refusing to dispatch {!r} [id={}] — "
                    "system/reminder tasks must declare 'Recipients:'. "
                    "Add Recipients to HEARTBEAT.md (e.g. "
                    "'Recipients: primary:whatsapp') and the task will "
                    "fire on the next tick.",
                    task_name, task_id or "<no-id>",
                )
                continue

            due.append(DueTask(
                name=task_name, task_type=task_type, schedule=schedule_str,
                model=model, pre_check=pre_check, recipients=recipients,
                id=task_id, prompt_file=prompt_file,
            ))

        return due

    @staticmethod
    def _compute_task_statuses(content: str, now: datetime) -> str:
        """Parse ## User Tasks section and compute due status in Python.

        Returns a formatted string describing which tasks are DUE NOW and which
        are not yet due, or "" if no tasks with Schedule fields are found.
        """
        # Schedule strings are naive; strip tzinfo for comparison
        now = now.replace(tzinfo=None)
        user_tasks_match = re.search(r"^## User Tasks\s*$", content, re.MULTILINE)
        if not user_tasks_match:
            return ""

        section_start = user_tasks_match.end()
        next_section_match = re.search(r"^## ", content[section_start:], re.MULTILINE)
        if next_section_match:
            section_end = section_start + next_section_match.start()
        else:
            section_end = len(content)
        section = content[section_start:section_end]

        task_blocks = re.split(r"\n(?=###\s)", section)
        lines = []

        for block in task_blocks:
            block = block.strip()
            if not block.startswith("###"):
                continue

            name_match = re.match(r"###\s+(.+)", block)
            if not name_match:
                continue
            task_name = name_match.group(1).strip()

            schedule_match = re.search(
                r"Schedule:\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)", block
            )
            if not schedule_match:
                continue
            schedule_str = schedule_match.group(1).strip()

            if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", schedule_str):
                try:
                    schedule_dt = datetime.strptime(schedule_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    continue
            else:
                try:
                    schedule_dt = datetime.strptime(schedule_str, "%Y-%m-%d")
                except ValueError:
                    continue

            until_match = re.search(r"Until:\s*(\d{4}-\d{2}-\d{2})", block)
            if until_match:
                try:
                    until_dt = datetime.strptime(until_match.group(1), "%Y-%m-%d")
                    if now > until_dt:
                        continue
                except ValueError:
                    pass

            effective_due, effective_due_str = _effective_due(block, schedule_dt, schedule_str)

            id_match = _ID_PAT.search(block)
            task_id = id_match.group(1) if id_match else None
            id_chunk = f" [id={task_id}]" if task_id else ""

            now_str = now.strftime("%Y-%m-%d %H:%M")
            if now >= effective_due:
                lines.append(
                    f"  - '{task_name}'{id_chunk} IS DUE NOW "
                    f"(scheduled {effective_due_str}, now is {now_str})"
                )
            else:
                lines.append(
                    f"  - '{task_name}'{id_chunk} is NOT due until {effective_due_str}"
                )

        if not lines:
            return ""

        return (
            "Python-computed task due status (authoritative — trust this over "
            "your own date math). When ticking or completing a task via your "
            "task management tool, pass the `id=` value (e.g. `t_a2b3c4d5`) "
            "— not the title (matching by title breaks if you paraphrase it):\n"
            + "\n".join(lines)
        )

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def _decide(self, content: str) -> tuple[str, str, list[DueTask]]:
        """Phase 1: determine whether any tasks are due.

        When last_run_tracking=True: fully deterministic, no LLM call.
        When last_run_tracking=False: falls back to LLM tool call.

        Returns (action, tasks_str, due_tasks) where action is 'skip' or 'run'.
        due_tasks is populated only when last_run_tracking=True.
        """
        from nanobot.utils.helpers import current_time_str
        now = self._now()

        if self.last_run_tracking:
            due = self._compute_due_tasks(content, now.replace(tzinfo=None), self.model_presets)
            if not due:
                return "skip", "", []
            summary = ", ".join(f"{t.name} ({t.task_type})" for t in due)
            # Decision-log line keeps a grep-friendly id alongside the type so
            # we can correlate execution with the originating block in
            # HEARTBEAT.md (the Piedmont reminder bug was hard to debug
            # because logs only had the LLM-paraphrased title).
            log_summary = ", ".join(
                f"{t.name} [{t.task_type}{',' + t.id if t.id else ''}]"
                for t in due
            )
            logger.debug("Heartbeat: {} due task(s) — {}", len(due), log_summary)
            return "run", summary, due

        # LLM fallback when last_run_tracking is disabled
        now_str = current_time_str(self.timezone)
        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": "You are a heartbeat agent. Call the heartbeat tool to report your decision."},
                {"role": "user", "content": (
                    f"Current Time: {now_str}\n\n"
                    "Review the following HEARTBEAT.md and decide whether there are tasks DUE NOW "
                    "(scheduled date/time has already passed or is within the next 5 minutes). "
                    "Tasks scheduled for a future date are NOT due — choose 'skip' for those. \n\n"
                    f"{content}"
                )},
            ],
            tools=_HEARTBEAT_TOOL,
            model=self.model,
        )

        if not response.should_execute_tools:
            if response.has_tool_calls:
                logger.warning(
                    "Ignoring heartbeat tool calls under finish_reason='{}'",
                    response.finish_reason,
                )
            return "skip", "", []

        args = response.tool_calls[0].arguments
        return args.get("action", "skip"), args.get("tasks", ""), []

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Heartbeat started (every {}s)", self.interval_s)

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat error")

    @staticmethod
    def _is_deliverable(response: str) -> bool:
        """Check if a heartbeat response is suitable for user delivery.

        Filters out two classes of bad output before the evaluator runs:

        1. **Finalization fallback** — the runner hit empty-response retries
           and produced a canned error message.  For heartbeat, empty output
           is a valid "nothing to report" outcome, not a failure.
        2. **Leaked reasoning** — the model reflected internal file names,
           decision logic, or meta-commentary instead of a user-facing report.
        """
        text = response.lower()

        # Runner finalization fallback
        if "couldn't produce a final answer" in text:
            return False

        # Leaked internal reasoning patterns
        leaked_patterns = [
            "heartbeat.md",
            "awareness.md",
            "judgment call:",
            "decision logic",
            "valid options are",
            "my instructions",
            "i am supposed to",
            "strict heartbeat interpretation",
        ]
        if any(pattern in text for pattern in leaked_patterns):
            return False

        return True

    @staticmethod
    async def _run_pre_check(command: str) -> bool:
        """Run a pre-check command. Returns True if the task has work to do.

        A task is skipped (returns False) when the command exits successfully
        and its output is empty, an empty JSON array '[]', or starts with 'SKIP'.
        Non-zero exit codes are treated as errors → proceed with LLM to be safe.
        """
        proc = None
        try:
            args = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                logger.warning("Pre-check exited {} for '{}' — proceeding with LLM",
                               proc.returncode, command)
                return True
            output = stdout.decode().strip() if stdout else ""
            if not output or output == "[]" or output.upper().startswith("SKIP"):
                return False
            return True
        except asyncio.TimeoutError:
            logger.warning("Pre-check timed out for '{}' — proceeding with LLM", command)
            if proc:
                proc.kill()
                await proc.communicate()
            return True
        except Exception as e:
            logger.warning("Pre-check failed for '{}': {} — proceeding with LLM", command, e)
            return True  # On error, proceed with LLM to be safe

    async def _filter_by_pre_checks(self, tasks: list[DueTask]) -> list[DueTask]:
        """Run pre-check commands and filter out tasks with no work to do.

        The task's pre_check field is a registry key, resolved to an actual
        command via self.pre_check_registry. Unknown keys are logged and skipped
        (task proceeds to LLM).
        """
        result = []
        for task in tasks:
            if not task.pre_check:
                result.append(task)
                continue
            command = self.pre_check_registry.get(task.pre_check)
            if not command:
                logger.warning("Heartbeat: unknown pre-check key '{}' for '{}' — proceeding with LLM",
                               task.pre_check, task.name)
                result.append(task)
                continue
            has_work = await self._run_pre_check(command)
            if has_work:
                result.append(task)
            else:
                logger.info("Heartbeat: pre-check skipped '{}' (no work)", task.name)
        return result

    def _advance_schedules(self, tasks: list[DueTask]) -> None:
        """Deterministically advance Schedule for executed recurring tasks.

        After a task executes, advance its Schedule past now by its Recur
        interval and write Last-run.  When the LLM has already pushed
        Schedule into the future (via --tick), the Schedule advance is
        skipped but Last-run is still bumped — without that, a stale
        Last-run would let the cadence check re-fire the task on the
        very next tick.

        Re-reads HEARTBEAT.md fresh to avoid overwriting changes made
        during task execution (which can take 10-30s). The whole
        read-modify-write window runs under ``heartbeat_lock`` so
        concurrent ``tasks_update.py --tick`` invocations from the LLM
        can't race the advance.

        Do not call while already holding ``heartbeat_lock`` — flock is
        non-reentrant and would deadlock.
        """
        with heartbeat_lock(self.workspace):
            self._advance_schedules_locked(tasks)

    def _advance_schedules_locked(self, tasks: list[DueTask]) -> None:
        content = self._read_heartbeat_file()
        if not content:
            return

        now_naive = self._now().replace(tzinfo=None)
        now_str = now_naive.strftime("%Y-%m-%d %H:%M")

        changed = False
        for task in tasks:
            if task.task_type == "announcement" or not task.schedule:
                continue

            # Recompute section bounds each iteration: a previous task in
            # this loop may have mutated content (Schedule/Last-run rewrite),
            # which shifts offsets within the file.
            #
            # Scope: everything up to ## Completed (exclusive). The only
            # section we MUST exclude is Completed — a task re-added after
            # being ticked could share an Id with a leftover entry there.
            # Including any other intermediate sections (e.g. ## System
            # Tasks) is safe because Ids are unique per active block.
            completed_match = re.search(r"^## Completed\s*$", content, re.MULTILINE)
            section_start = 0
            section_end = completed_match.start() if completed_match else len(content)

            # Resolve the block as (block_start, block_end) within content.
            # Prefer id-based lookup — it survives LLM-paraphrased task names
            # (the original Piedmont reminder bug). Fall back to name match
            # for legacy blocks rolled out before tasks_update.py started
            # writing Id lines.
            block_start = block_end = -1

            if task.id:
                id_line_pat = re.compile(
                    rf"^Id:\s*{re.escape(task.id)}\s*$", re.MULTILINE,
                )
                id_m = id_line_pat.search(content, section_start, section_end)
                if id_m:
                    # Constrain walk-back to the enclosing ## section so
                    # we don't latch onto a `### ` heading from a previous
                    # section (e.g. preamble notes above ## User Tasks).
                    sec_nl = content.rfind("\n## ", section_start, id_m.start())
                    walk_start = sec_nl + 1 if sec_nl != -1 else section_start
                    # Walk back to the enclosing `### ` heading.
                    nl = content.rfind("\n### ", walk_start, id_m.start())
                    if nl != -1:
                        block_start = nl + 1  # skip the leading newline
                    elif content.startswith("### ", walk_start):
                        block_start = walk_start
                    if block_start != -1:
                        rest = content[block_start:section_end]
                        end_m = re.search(r"\n###\s|\n##\s", rest)
                        block_end = block_start + (end_m.start() if end_m else len(rest))

            if block_start == -1 and not task.id:
                escaped = re.escape(task.name)
                block_pat = re.compile(
                    rf"(###\s+{escaped}\s*\n)(.*?)(?=\n###\s|\n##\s|\Z)",
                    re.DOTALL,
                )
                m = block_pat.search(content, section_start, section_end)
                if m:
                    block_start, block_end = m.start(), m.end()

            if block_start == -1:
                if task.id:
                    logger.warning(
                        "Heartbeat: could not find block for '{}' [id={}] to advance schedule",
                        task.name, task.id,
                    )
                else:
                    logger.warning(
                        "Heartbeat: could not find block for '{}' to advance schedule",
                        task.name,
                    )
                continue

            block = content[block_start:block_end]

            recur_m = _RECUR_PAT.search(block)
            if not recur_m:
                continue

            recur_n = int(recur_m.group(1)) or 1  # treat 0 as 1 to prevent infinite loops
            recur_unit = recur_m.group(2).lower()

            sched_m = _SCHED_PAT.search(block)
            if not sched_m:
                continue
            schedule_str = sched_m.group(1).strip()

            try:
                if " " in schedule_str:
                    current_dt = datetime.strptime(schedule_str, "%Y-%m-%d %H:%M")
                    has_time = True
                else:
                    current_dt = datetime.strptime(schedule_str, "%Y-%m-%d")
                    has_time = False
            except ValueError:
                continue

            # Already-future Schedule: skip the bump, but still write
            # Last-run so the cadence check doesn't immediately re-fire
            # this task on the next tick.
            if current_dt > now_naive:
                logger.info("Heartbeat: '{}' schedule already advanced, bumping Last-run", task.name)
                if _LASTRUN_PAT.search(block):
                    updated_block = _LASTRUN_PAT.sub(f"Last-run: {now_str}", block, count=1)
                else:
                    updated_block = re.sub(
                        r"(Schedule:[^\n]+)(\n|$)",
                        rf"\1\nLast-run: {now_str}\2",
                        block,
                        count=1,
                    )
                if updated_block != block:
                    content = content[:block_start] + updated_block + content[block_end:]
                    changed = True
                continue

            next_dt = _apply_recur(current_dt, recur_n, recur_unit)
            if next_dt is None:
                # Unknown unit slipped past _RECUR_PAT — refuse to silently
                # advance by a guessed delta (the regression that re-fired
                # `every 1 month` tasks every tick because the unit had no
                # branch here).
                continue
            # For variable-length units (month) the modulo trick doesn't
            # work, so step until we're past now_naive. Bounded by the
            # iteration count of (now - current_dt) / unit, which is tiny
            # for any realistic clock skew.
            while next_dt <= now_naive:
                stepped = _apply_recur(next_dt, recur_n, recur_unit)
                if stepped is None or stepped <= next_dt:
                    break  # defensive: avoid infinite loop on bad input
                next_dt = stepped

            if recur_unit in ("minute", "hour") or has_time:
                next_str = next_dt.strftime("%Y-%m-%d %H:%M")
            else:
                next_str = next_dt.strftime("%Y-%m-%d")

            # Note: Until date enforcement is handled by _compute_due_tasks,
            # not here. Always advance the schedule to prevent infinite loops
            # when now < Until < next_dt.

            updated_block = re.sub(
                r"(Schedule:\s*)" + re.escape(schedule_str),
                rf"\g<1>{next_str}",
                block,
                count=1,
            )
            if _LASTRUN_PAT.search(updated_block):
                updated_block = _LASTRUN_PAT.sub(f"Last-run: {now_str}", updated_block, count=1)
            else:
                updated_block = re.sub(
                    r"(Schedule:[^\n]+)(\n|$)",
                    rf"\1\nLast-run: {now_str}\2",
                    updated_block,
                    count=1,
                )

            content = content[:block_start] + updated_block + content[block_end:]
            changed = True
            id_chunk = f" [id={task.id}]" if task.id else ""
            logger.info(
                "Heartbeat: advanced '{}'{} schedule to {}",
                task.name, id_chunk, next_str,
            )

        if changed:
            write_text_atomic(self.heartbeat_file, content)

    def _read_prompt_file(self, raw_path: str, recipient: str | None) -> str | None:
        """Resolve and read a task's Prompt-file, substituting {recipient}.

        - `{recipient}` placeholders are replaced with the supplied recipient
          name. If the path contains `{recipient}` but none is supplied,
          returns None (caller falls back to the default task summary).
        - Paths are resolved relative to the heartbeat workspace and must
          stay under it — any traversal outside the workspace returns None
          (defense against a malformed task block escaping the sandbox).
        - Returns None on any I/O failure; the dispatcher falls back to the
          task-summary path so a missing/unreadable file doesn't silently
          drop the task.
        """
        path_str = raw_path
        if "{recipient}" in path_str:
            if not recipient:
                logger.warning(
                    "Heartbeat: prompt-file {!r} references {{recipient}} "
                    "but task has no Recipients — skipping prompt file",
                    raw_path,
                )
                return None
            path_str = path_str.replace("{recipient}", recipient)

        candidate = (self.workspace / path_str).resolve()
        try:
            workspace_resolved = self.workspace.resolve()
            candidate.relative_to(workspace_resolved)
        except ValueError:
            # Log the post-substitution path too — when the brief silently
            # degrades, the substituted recipient name is the most common
            # culprit (typo, missing user file) and the raw path alone hides
            # it.
            logger.warning(
                "Heartbeat: prompt-file {!r} (resolved {!r}) escapes the workspace, refusing",
                raw_path, str(candidate),
            )
            return None

        try:
            return candidate.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(
                "Heartbeat: prompt-file {!r} unreadable ({}); falling back to summary",
                str(candidate), e,
            )
            return None

    async def _dispatch_task_to_recipients(self, task: DueTask, evaluate_response) -> None:
        """Run a single task, fanning out per recipient.

        Each ``(recipient_symbol, channel)`` pair from ``Recipients:`` is
        resolved via the host's ``users_loader`` to a channel handle and
        the resolved ``(channel, handle)`` is passed to ``on_execute`` /
        ``on_notify`` as ``target=``. The agent runs in the recipient's
        session so ``message()`` defaults route correctly — the LLM does
        not have to (and must not) guess chat_id from memory.

        Message body per dispatch:
          - Prompt-file tasks: read the file with ``{recipient}``
            substitution (per-user prompt content).
          - Other tasks: ``"<name> (<task_type>)"`` summary.

        Recipients are required. Without them there is no safe target,
        so the parser refuses to enqueue such tasks (see Rule 3 in
        ``_compute_due_tasks``); this method asserts the invariant
        belt-and-suspenders.

        Resolution failures skip that recipient. The schedule advances
        only when at least one delivery succeeded — a task whose
        Recipients are all unresolvable retries on the next tick rather
        than silently advancing past the failure.
        """
        pairs = task.recipient_pairs()
        if not pairs:
            # Should be unreachable — _compute_due_tasks refuses to enqueue.
            logger.error(
                "Heartbeat: {!r} reached dispatch without Recipients; refusing. "
                "(Parser invariant violated — check _compute_due_tasks.)",
                task.name,
            )
            return

        any_resolved = False  # at least one recipient resolved to a target
        for recipient, channel in pairs:
            target = _resolve_dispatch_target(recipient, channel)
            if target is None:
                logger.warning(
                    "Heartbeat: {!r} could not resolve recipient {!r}:{!r}; skipping.",
                    task.name, recipient, channel,
                )
                continue
            any_resolved = True

            if task.prompt_file:
                message = self._read_prompt_file(task.prompt_file, recipient)
                if message is None:
                    # Missing prompt-file still dispatches — a typo in the path
                    # shouldn't silently disable the task. Use the task summary.
                    message = f"{task.name} ({task.task_type})"
            else:
                message = f"{task.name} ({task.task_type})"

            # Pass `target` to the context hook so the host (homer) can pin
            # MessageTool.allowed_recipients to exactly this recipient — the
            # hard guarantee that the LLM can't override chat_id mid-turn.
            ctx = (
                self.on_execute_context([task], target=target)
                if self.on_execute_context
                else nullcontext()
            )
            try:
                with ctx:
                    response = await self.on_execute(
                        message, task.model, target=target,
                    )
                if response:
                    should_notify = await evaluate_response(
                        response, task.name, self.provider, self.model,
                        suppress_errors=self.suppress_errors,
                    )
                    if should_notify and self.on_notify:
                        logger.info(
                            "Heartbeat: completed, delivering response "
                            "(task={!r}, recipient={!r})",
                            task.name, recipient,
                        )
                        await self.on_notify(response, target=target)
                    else:
                        logger.info("Heartbeat: silenced by post-run evaluation")
            except Exception:
                logger.exception(
                    "Heartbeat: task failed for {!r} (recipient={!r})",
                    task.name, recipient,
                )

        if not any_resolved:
            # No recipient resolved — config bug (e.g., symbol missing from
            # users.yaml). Don't advance: this is the case where retrying
            # next tick has a chance of succeeding (operator may be fixing
            # users.yaml right now). Distinguished from per-recipient
            # on_execute failures, which DO advance (those are typically
            # persistent LLM errors and retrying every tick spams).
            logger.error(
                "Heartbeat: {!r} had Recipients but none resolved — schedule NOT "
                "advanced; will retry next tick. Check users.yaml for the "
                "configured symbols.", task.name,
            )
            return
        if self.last_run_tracking:
            self._advance_schedules([task])

    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        from nanobot.utils.evaluator import evaluate_response

        content = self._read_heartbeat_file()
        if not content:
            logger.debug("Heartbeat: HEARTBEAT.md missing or empty")
            return

        logger.info("Heartbeat: checking for tasks...")

        try:
            action, tasks_str, due_tasks = await self._decide(content)

            if action != "run":
                logger.info("Heartbeat: OK (nothing to report)")
                return

            # Run pre-checks to filter out tasks with no work
            # Only applies when last_run_tracking=True (structured due_tasks).
            # LLM fallback path (due_tasks empty) reads raw HEARTBEAT.md and
            # has no awareness of pre-check results.
            if self.last_run_tracking and due_tasks:
                due_tasks = await self._filter_by_pre_checks(due_tasks)
                if not due_tasks:
                    logger.info("Heartbeat: all tasks skipped by pre-checks")
                    return

            logger.info("Heartbeat: tasks found, executing...")
            if self.on_execute:
                # If no structured tasks (LLM fallback), run as before
                if not due_tasks:
                    # Tag the LLM call as heartbeat_system + synthetic so the
                    # $ai_generation event lands on the right dashboard row.
                    # Without this wrap the fallback would emit task_kind=chat
                    # because the LLMProvider has no contextvar set. (#54)
                    from nanobot.analytics.llm_telemetry import llm_telemetry_context

                    with llm_telemetry_context(
                        task_kind="heartbeat_system", is_synthetic=True,
                    ):
                        response = await self.on_execute(tasks_str, None)
                    if response and self._is_deliverable(response):
                        should_notify = await evaluate_response(
                            response, tasks_str, self.provider, self.model,
                            suppress_errors=self.suppress_errors,
                        )
                        if should_notify and self.on_notify:
                            logger.info("Heartbeat: completed, delivering response")
                            await self.on_notify(response)
                        else:
                            logger.info("Heartbeat: silenced by post-run evaluation")
                    elif response:
                        logger.info(
                            "Heartbeat: suppressed non-deliverable response ({})",
                            response[:80],
                        )
                else:
                    # Every system/reminder task dispatches per recipient
                    # via _dispatch_task_to_recipients, which passes
                    # target=(channel, handle) so the agent's message()
                    # tool routes to the correct user without guessing.
                    # Rule 3 (in _compute_due_tasks) guarantees every
                    # system/reminder task has Recipients before reaching
                    # dispatch.
                    #
                    # Announcements are the only remaining legacy-path
                    # dispatch — they have no recipients (they're
                    # broadcast updates curated at container startup) and
                    # fall through to on_execute() without a target. If
                    # we ever LLM-generate announcement content, add
                    # Recipients to that path too.
                    recipient_tasks = [t for t in due_tasks if t.recipients]
                    broadcast_tasks = [t for t in due_tasks if not t.recipients]

                    for task in recipient_tasks:
                        await self._dispatch_task_to_recipients(task, evaluate_response)

                    if broadcast_tasks:
                        # Group by model override (legacy path — announcements only).
                        groups: dict[str | None, list[DueTask]] = {}
                        for t in broadcast_tasks:
                            groups.setdefault(t.model, []).append(t)

                        for model_override, group_tasks in groups.items():
                            summary = ", ".join(f"{t.name} ({t.task_type})" for t in group_tasks)
                            ctx = (
                                self.on_execute_context(group_tasks)
                                if self.on_execute_context
                                else nullcontext()
                            )
                            try:
                                with ctx:
                                    response = await self.on_execute(summary, model_override)
                                if response and self._is_deliverable(response):
                                    should_notify = await evaluate_response(
                                        response, summary, self.provider, self.model,
                                        suppress_errors=self.suppress_errors,
                                    )
                                    if should_notify and self.on_notify:
                                        logger.info("Heartbeat: completed, delivering response")
                                        await self.on_notify(response)
                                    else:
                                        logger.info("Heartbeat: silenced by post-run evaluation")
                                elif response:
                                    logger.info(
                                        "Heartbeat: suppressed non-deliverable response ({})",
                                        response[:80],
                                    )
                            except Exception:
                                logger.exception("Heartbeat: task failed for {}", summary)
                            finally:
                                # Always advance schedule — even on failure — to prevent
                                # retry spam on persistent errors (e.g. API outage).
                                if self.last_run_tracking:
                                    self._advance_schedules(group_tasks)
        except Exception:
            logger.exception("Heartbeat execution failed")

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat."""
        content = self._read_heartbeat_file()
        if not content:
            return None
        action, tasks_str, due_tasks = await self._decide(content)
        if action != "run" or not self.on_execute:
            return None

        # If structured tasks available, group by model and run each group
        if due_tasks:
            groups: dict[str | None, list[DueTask]] = {}
            for t in due_tasks:
                groups.setdefault(t.model, []).append(t)
            results: list[str] = []
            for model_override, group_tasks in groups.items():
                summary = ", ".join(f"{t.name} ({t.task_type})" for t in group_tasks)
                result = await self.on_execute(summary, model_override)
                if result:
                    results.append(result)
            return "\n".join(results) if results else None

        return await self.on_execute(tasks_str, None)
