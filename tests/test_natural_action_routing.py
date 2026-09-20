"""Natural requests retain their wording despite similar installed commands."""

from dataclasses import replace

import pytest

from core.capabilities import capabilities
from core.command_resolver import CommandResolver, IntuitionCommandProvider, StaticCommandProvider
from core.os_intents import _try_os_intent, is_app_command
from interface import server
from test_hud_resolution import hud as hud
from test_hud_resolution import preview, receive, submit


SHELL_NAMES = ["openai", "check-node", "write", "show", "find", "tell", "git", "echo", "python"]
NATURAL_REQUESTS = [
    "open the chrome and check the weather for me",
    "check the weather for me",
    "open chrome and tell me what time it is",
    "openchrome for me and check the weather",
    "Please open my browser and find today's forecast",
    "go to chrome and check the weather",
    "show me the forecast for tomorrow",
    "find my recent notes",
    "tell me what time it is",
    "Could you please check my calendar for tomorrow?",
    "check the battery and write a note about it",
    "open the chrome and take a screenshot",
]


@pytest.fixture
def resolver():
    return CommandResolver([
        IntuitionCommandProvider(), StaticCommandProvider(SHELL_NAMES, case_sensitive=False),
    ], shell="cmd")


@pytest.mark.parametrize("text", NATURAL_REQUESTS)
def test_complete_natural_request_is_never_corrected_to_an_executable(resolver, text):
    assert is_app_command(text)
    assert _try_os_intent(text) is None
    resolution = resolver.resolve(text)
    assert resolution.original == text
    assert resolution.status == "exact"
    assert resolution.namespace == "intuitionos"
    assert resolution.candidates == []


@pytest.mark.parametrize("text", [
    "open the chrome and check the weather for me",
    "check the weather for me",
    "open the chrome and take a screenshot",
    "check the battery and write a note about it",
])
def test_hud_passes_the_complete_original_instruction_to_the_model(hud, monkeypatch, text):
    client, _ = hud
    server._state["resolver"].providers.append(StaticCommandProvider(SHELL_NAMES, case_sensitive=False))
    calls = []

    def step(original, **kwargs):
        calls.append(original)
        return {"reply": "Received the entire instruction."}

    monkeypatch.setattr(server._state["brain"], "step", step)
    for name in ("os_open_app", "os_open_url", "os_take_screenshot", "os_get_battery", "run_command"):
        monkeypatch.setitem(capabilities._caps, name, replace(
            capabilities.get(name), fn=lambda **kwargs: pytest.fail("A partial action or shell command ran"),
        ))
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, text)
        assert not shown["candidates"] and shown["namespace"] == "intuitionos"
        submit(ws, shown, index=None)
        assert "entire instruction" in receive(ws, "reply")["text"]
    assert calls == [text]


@pytest.mark.parametrize("text", [
    'openai chat "check the weather for me"',
    'check-node --message "check the weather for me"',
    'git commit -m "open chrome and check the weather"',
    'echo "show me the forecast"',
    'python script.py --text "find my notes"',
    'find /C "the weather" notes.txt',
    "find my.txt", "show --help", "tell --version", "write notes.txt",
])
def test_real_shell_heads_keep_their_opaque_arguments(resolver, text):
    assert not is_app_command(text)
    resolution = resolver.resolve(text)
    assert resolution.original == text
    assert resolution.status == "exact"
    assert resolution.namespace == ("git" if text.startswith("git ") else "shell")
    assert resolution.candidates == []


def test_explicit_exec_still_allows_shell_name_correction(resolver):
    text = "/exec check the weather for me"
    assert not is_app_command(text)
    resolution = resolver.resolve(text)
    assert resolution.namespace == "shell"
    assert resolution.candidates[0].text == "/exec check-node the weather for me"


@pytest.mark.parametrize("text,capability", [
    ("open chrome", "os_open_app"),
    ("open chrome and go to google.com", "os_open_url"),
    ("check the battery", "os_get_battery"),
    ("show the current volume", "os_get_volume"),
])
def test_supported_single_actions_still_use_existing_direct_tools(text, capability):
    assert is_app_command(text)
    assert _try_os_intent(text)[0] == capability


# ── Arranging windows out loud ──────────────────────────────────────────────
#
# Said aloud this is the most natural thing to ask for and the least worth
# waking a model over, so it is recognised directly and works with Ollama down.


@pytest.mark.parametrize("said,expected", [
    ("snap this window to the left",      ("os_snap_window", {"position": "left"})),
    ("move this window to the right",     ("os_snap_window", {"position": "right"})),
    ("put it on the left",                ("os_snap_window", {"position": "left"})),
    ("snap left",                         ("os_snap_window", {"position": "left"})),
    ("move this window to the top right", ("os_snap_window", {"position": "top-right"})),
    ("dock this window bottom left",      ("os_snap_window", {"position": "bottom-left"})),
    ("maximize this window",              ("os_window_state", {"state": "maximize"})),
    ("minimise",                          ("os_window_state", {"state": "minimize"})),
    ("restore this window",               ("os_window_state", {"state": "restore"})),
    ("next window",                       ("os_cycle_window", {"direction": "next"})),
    ("switch to the previous window",     ("os_cycle_window", {"direction": "previous"})),
    ("list my open windows",              ("os_list_windows", {})),
])
def test_window_arrangement_is_recognised_without_the_model(said, expected):
    assert _try_os_intent(said) == expected


@pytest.mark.parametrize("said", [
    "move file.txt to backup/",       # a real file operation, not a window
    "minimize the risk of failure",   # prose that happens to start with a verb
    "git push",
    "python train.py --lr 0.001",
    "left",                           # a bare word is not an instruction
    "what is the weather",
])
def test_ordinary_input_is_not_mistaken_for_a_window_command(said):
    """A false positive here would move a window instead of running a command."""
    routed = _try_os_intent(said)
    assert routed is None or not routed[0].startswith(
        ("os_snap_window", "os_window_state", "os_cycle_window", "os_list_windows")
    ), f"{said!r} was hijacked into {routed}"


def test_a_spoken_window_command_is_reversible_and_passes_the_gate():
    from core.actions import register_os_capabilities
    from core.capabilities import capabilities, gate

    register_os_capabilities()
    name, args = _try_os_intent("snap this window to the left")
    decision = gate(capabilities.get(name), dict(args), actor="user", confidence=1.0)
    assert decision.verdict == "allow"
    assert capabilities.get(name).reversibility == "reversible"
