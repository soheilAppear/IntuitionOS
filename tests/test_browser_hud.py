"""Browser requests use the real HUD route and gate without asking a model."""

from dataclasses import replace

import pytest

from core.actions import actions
from core.capabilities import capabilities, is_safe_mode, set_safe_mode
from core.command_resolver import StaticCommandProvider
from interface import server
from test_hud_resolution import hud as hud
from test_hud_resolution import preview, receive, submit


@pytest.mark.parametrize("text,browser", [
    ("open chrome and go to google.com", "chrome"),
    ("go to chrome open it and go to google.com", "chrome"),
    ("open google.com in chrome", "chrome"),
    ("openchrome for me and go to google.com", "chrome"),
    ("go to google.com", "default"),
])
def test_browser_request_finishes_without_model_or_disabling_safe_mode(hud, monkeypatch, text, browser):
    client, _ = hud
    set_safe_mode(True)
    calls = []
    server._state["resolver"].providers.append(StaticCommandProvider(
        ["openai", "openssl", "openclaw", "go", "goto"], case_sensitive=False,
    ))

    def open_url(**args):
        calls.append(args)
        return {"ok": True, "url": args["url"], "browser": args["browser"], "status": "open_requested"}

    monkeypatch.setitem(capabilities._caps, "os_open_url", replace(
        capabilities.get("os_open_url"), fn=open_url,
    ))
    monkeypatch.setattr(server._state["brain"], "step", lambda *a, **k: pytest.fail("Browser request reached the model"))
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, text)
        assert not shown["candidates"]
        assert shown["namespace"] == "intuitionos"
        submit(ws, shown, index=None)
        assert receive(ws, "thinking")["text"] == "Opening browser…"
        reply = receive(ws, "reply")["text"]
        assert "https://google.com" in reply
        assert ("Chrome" if browser == "chrome" else "default browser") in reply
        assert calls == [{"url": "https://google.com", "browser": browser}]
        assert is_safe_mode()
        assert server.get_journal().recent()[0]["capability"] == "os_open_url"


def test_browser_launch_failure_returns_a_clear_reply(hud, monkeypatch):
    client, _ = hud
    monkeypatch.setitem(capabilities._caps, "os_open_url", replace(
        capabilities.get("os_open_url"), fn=lambda **args: {"error": "Chrome was not found. Install Chrome or use your default browser."},
    ))
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        submit(ws, preview(ws, "open google.com in chrome"), index=None)
        assert "Chrome was not found" in receive(ws, "reply")["text"]


@pytest.mark.parametrize("args", [
    {"url": "javascript:alert(1)"},
    {"url": "file:///C:/Windows/System32/cmd.exe"},
    {"url": "https://google.com", "browser": "cmd.exe"},
])
def test_browser_gate_rejects_non_web_targets_before_launch(hud, monkeypatch, args):
    monkeypatch.setitem(capabilities._caps, "os_open_url", replace(
        capabilities.get("os_open_url"), fn=lambda **kwargs: pytest.fail("Invalid request reached browser"),
    ))
    assert actions.dispatch("os_open_url", args)["denied"]


def test_browser_open_cannot_be_speculatively_prewarmed(hud):
    result = actions.dispatch("os_open_url", {"url": "https://google.com"}, actor="anticipator")
    assert result["denied"]
