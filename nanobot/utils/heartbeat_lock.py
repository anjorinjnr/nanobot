"""Cross-process advisory locking for HEARTBEAT.md read-modify-write.

Two writers can collide on HEARTBEAT.md: the heartbeat service inside the
agent process (``_advance_schedules``) and external CLI tools spawned via
``exec`` (e.g. Homer's ``tasks_update.py --tick``). Without coordination
the slower of two interleaved RMW cycles overwrites the other's update.

Callers wrap the entire read-modify-write window:

    from nanobot.utils.helpers import write_text_atomic

    with heartbeat_lock(workspace):
        content = (workspace / "HEARTBEAT.md").read_text()
        new = mutate(content)
        if new != content:
            write_text_atomic(workspace / "HEARTBEAT.md", new)

The lock is held on a sibling ``.heartbeat.lock`` file so it works before
HEARTBEAT.md exists and survives the atomic-rename write step.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILENAME = ".heartbeat.lock"


@contextmanager
def heartbeat_lock(workspace: Path | str) -> Iterator[None]:
    """Acquire an exclusive flock for the workspace's HEARTBEAT.md.

    Not reentrant: nesting two ``with heartbeat_lock(...)`` blocks in the
    same process deadlocks (each block opens a separate fd, and flock is
    per open file description). Callers that need to chain mutations must
    do so within a single block, or release between operations.
    """
    lock_path = Path(workspace) / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
