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
