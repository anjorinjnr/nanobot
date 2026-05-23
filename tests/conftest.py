"""Shared pytest fixtures for nanobot tests.

Mostly: making the homer sibling clone available to tests that exercise
homer-specific code paths (channel auto-heal, prompt-file dispatch). The
homer repo lives at ``../homer`` relative to nanobot in CI/dev.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HOMER_TOOLS = Path(__file__).resolve().parent.parent.parent / "homer" / "tools"


@pytest.fixture
def homer_users_yaml(tmp_path, monkeypatch):
    """Provision an empty tmp users.yaml + put homer's tools on sys.path so
    homer's ``users_loader`` is importable by code under test. Tests that
    need pre-seeded content write to the returned path themselves.

    Skips when the homer sibling clone isn't present — tests using this
    fixture assert behavior of homer↔nanobot interop, not nanobot alone.
    """
    if not _HOMER_TOOLS.exists():
        pytest.skip(f"homer/tools not at {_HOMER_TOOLS} (expected sibling clone)")
    monkeypatch.syspath_prepend(str(_HOMER_TOOLS))
    # The lazy import in the code under test may have a stale users_loader
    # cached from a previous test that pointed at a different HOMER_USERS_YAML.
    sys.modules.pop("users_loader", None)
    path = tmp_path / "users.yaml"
    monkeypatch.setenv("HOMER_USERS_YAML", str(path))
    return path
