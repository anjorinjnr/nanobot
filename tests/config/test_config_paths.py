from pathlib import Path

from nanobot.config.paths import (
    get_cli_history_path,
    get_cron_dir,
    get_data_dir,
    get_legacy_sessions_dir,
    get_logs_dir,
    get_media_dir,
    get_persistent_data_dir,
    get_runtime_subdir,
    get_workspace_path,
    is_default_workspace,
)


def test_runtime_dirs_follow_config_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-a" / "config.json"
    monkeypatch.setattr("nanobot.config.paths.get_config_path", lambda: config_file)

    assert get_data_dir() == config_file.parent
    assert get_runtime_subdir("cron") == config_file.parent / "cron"
    assert get_cron_dir() == config_file.parent / "cron"
    assert get_logs_dir() == config_file.parent / "logs"


def test_media_dir_supports_channel_namespace(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-b" / "config.json"
    monkeypatch.setattr("nanobot.config.paths.get_config_path", lambda: config_file)

    assert get_media_dir() == config_file.parent / "media"
    assert get_media_dir("telegram") == config_file.parent / "media" / "telegram"


def test_shared_and_legacy_paths_remain_global() -> None:
    assert get_cli_history_path() == Path.home() / ".nanobot" / "history" / "cli_history"
    assert get_legacy_sessions_dir() == Path.home() / ".nanobot" / "sessions"


def test_workspace_path_is_explicitly_resolved() -> None:
    assert get_workspace_path() == Path.home() / ".nanobot" / "workspace"
    assert get_workspace_path("~/custom-workspace") == Path.home() / "custom-workspace"


def test_is_default_workspace_distinguishes_default_and_custom_paths() -> None:
    assert is_default_workspace(None) is True
    assert is_default_workspace(Path.home() / ".nanobot" / "workspace") is True
    assert is_default_workspace("~/custom-workspace") is False


def test_persistent_data_dir_defaults_to_data_dir(monkeypatch, tmp_path: Path) -> None:
    """Without the env var, persistent dir == data dir — backward compatible
    with bare-metal / systemd deployments where data_dir already persists."""
    monkeypatch.delenv("NANOBOT_PERSISTENT_DATA_DIR", raising=False)
    config_file = tmp_path / "instance-c" / "config.json"
    monkeypatch.setattr("nanobot.config.paths.get_config_path", lambda: config_file)
    assert get_persistent_data_dir() == get_data_dir()


def test_persistent_data_dir_honors_env_override(monkeypatch, tmp_path: Path) -> None:
    """Hosted deployments set NANOBOT_PERSISTENT_DATA_DIR to a mounted
    volume so state like lid_map.json survives container recreation."""
    persistent = tmp_path / "persistent"
    monkeypatch.setenv("NANOBOT_PERSISTENT_DATA_DIR", str(persistent))
    assert get_persistent_data_dir() == persistent
    # Directory is created on first call (ensure_dir).
    assert persistent.is_dir()


def test_persistent_data_dir_expands_tilde(monkeypatch) -> None:
    """User-provided paths with ~ should resolve to the home directory."""
    monkeypatch.setenv("NANOBOT_PERSISTENT_DATA_DIR", "~/nanobot-persistent")
    assert get_persistent_data_dir() == Path.home() / "nanobot-persistent"
