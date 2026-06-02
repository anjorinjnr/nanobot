"""Per-turn multimodal model routing (AgentLoop._select_dispatch_model).

A cheap text-only default (e.g. deepseek) should keep handling text turns,
while turns that actually carry image/audio content route to a configured
vision model — and only when the two models differ.
"""

from types import SimpleNamespace

from nanobot.agent.loop import AgentLoop


# ── _messages_have_multimodal ──────────────────────────────────────────────

def test_string_content_is_text():
    assert AgentLoop._messages_have_multimodal([{"role": "user", "content": "hi"}]) is False


def test_text_only_blocks_are_text():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert AgentLoop._messages_have_multimodal(msgs) is False


def test_image_url_block_is_multimodal():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "what's this?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    ]}]
    assert AgentLoop._messages_have_multimodal(msgs) is True


def test_audio_block_is_multimodal():
    msgs = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]
    assert AgentLoop._messages_have_multimodal(msgs) is True


def test_empty_and_none_are_text():
    assert AgentLoop._messages_have_multimodal([]) is False
    assert AgentLoop._messages_have_multimodal(None) is False


def test_non_dict_messages_skipped():
    assert AgentLoop._messages_have_multimodal(["junk", None, 42]) is False


# ── _select_dispatch_model ─────────────────────────────────────────────────

def _select(model, multimodal_model, messages):
    """Invoke the instance method against a duck-typed loop."""
    obj = SimpleNamespace(
        model=model,
        multimodal_model=multimodal_model,
        _messages_have_multimodal=AgentLoop._messages_have_multimodal,
    )
    return AgentLoop._select_dispatch_model(obj, messages)


_IMG = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
_TXT = [{"role": "user", "content": "hello"}]


def test_routes_multimodal_turn_to_vision_model():
    assert _select("deepseek/deepseek-v4-flash", "google/gemini-3-flash-preview", _IMG) \
        == "google/gemini-3-flash-preview"


def test_text_turn_stays_on_default_even_with_vision_configured():
    assert _select("deepseek/deepseek-v4-flash", "google/gemini-3-flash-preview", _TXT) \
        == "deepseek/deepseek-v4-flash"


def test_no_vision_model_configured_keeps_default_on_image():
    # Graceful: nothing to route to — keep the default; the provider's
    # strip-and-retry fallback handles the unsupported image.
    assert _select("deepseek/deepseek-v4-flash", None, _IMG) == "deepseek/deepseek-v4-flash"


def test_identical_models_no_pointless_switch():
    same = "google/gemini-3-flash-preview"
    assert _select(same, same, _IMG) == same


def test_vision_model_used_only_for_the_multimodal_turn():
    m = "deepseek/deepseek-v4-flash"
    v = "google/gemini-3-flash-preview"
    assert _select(m, v, _TXT) == m   # text → cheap default
    assert _select(m, v, _IMG) == v   # image → vision


def test_cli_direct_agentloop_constructions_forward_multimodal_model():
    """Regression guard: the gateway/serve commands build AgentLoop directly
    (not via from_config), so each direct construction MUST forward
    multimodal_model — otherwise per-turn routing silently no-ops in the
    gateway, which is the path homer actually runs.
    """
    import pathlib
    import nanobot.cli.commands as cmds

    src = pathlib.Path(cmds.__file__).read_text(encoding="utf-8")
    # `AgentLoop(` matches only direct constructions; `AgentLoop.from_config(`
    # contains `AgentLoop.`, not `AgentLoop(`.
    n_direct = src.count("AgentLoop(")
    n_multimodal = src.count("multimodal_model=")
    assert n_direct >= 1, "expected at least one direct AgentLoop construction"
    assert n_multimodal >= n_direct, (
        f"{n_direct} direct AgentLoop() construction(s) but only {n_multimodal} "
        "multimodal_model= kwarg(s) — a construction is dropping multimodal_model, "
        "so per-turn vision routing will no-op there."
    )
