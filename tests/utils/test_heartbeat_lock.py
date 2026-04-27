"""Tests for nanobot.utils.heartbeat_lock."""
from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from nanobot.utils.heartbeat_lock import LOCK_FILENAME, heartbeat_lock


def test_lock_creates_lockfile_in_workspace(tmp_path: Path) -> None:
    with heartbeat_lock(tmp_path):
        assert (tmp_path / LOCK_FILENAME).exists()


def test_lock_creates_workspace_dir_if_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "nested" / "ws"
    with heartbeat_lock(workspace):
        assert (workspace / LOCK_FILENAME).exists()


def _hold_lock(workspace: str, hold_s: float, ready_path: str, done_path: str) -> None:
    with heartbeat_lock(workspace):
        Path(ready_path).write_text("ready")
        time.sleep(hold_s)
        Path(done_path).write_text("done")


def test_lock_serializes_across_processes(tmp_path: Path) -> None:
    # The whole point of the lock: two processes RMW HEARTBEAT.md at the
    # same time and the second waits for the first. Models nanobot's
    # _advance_schedules racing against an LLM-spawned tasks_update.py.
    ready = tmp_path / "ready"
    done = tmp_path / "done"
    proc = multiprocessing.Process(
        target=_hold_lock, args=(str(tmp_path), 0.4, str(ready), str(done))
    )
    proc.start()
    try:
        deadline = time.monotonic() + 2.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "child failed to acquire lock"

        t0 = time.monotonic()
        with heartbeat_lock(tmp_path):
            elapsed = time.monotonic() - t0
        # Child held for 0.4s; we entered ready ~immediately, so we
        # should have blocked for almost the full hold period.
        assert elapsed > 0.2, f"lock did not block (elapsed {elapsed}s)"
        assert done.read_text() == "done"
    finally:
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()
