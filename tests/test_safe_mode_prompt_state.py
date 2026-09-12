"""Mode-change permission follows current mode without replacing action approval."""

from dataclasses import replace
import json

import pytest

from core.capabilities import capabilities, is_safe_mode, set_safe_mode
from interface import server
from test_hud_resolution import hud as hud  # Shared real WebSocket fixture.
from test_hud_resolution import preview, receive, submit


@pytest.fixture
def command_calls(monkeypatch):
    calls = []
    monkeypatch.setitem(capabilities._caps, "run_command", replace(
        capabilities.get("run_command"),
        fn=lambda **args: calls.append(args) or {"returncode": 0, "stdout": "completed"},
    ))
    return calls


@pytest.mark.parametrize("actor", ["user", "model"])
def test_request_while_off_only_requests_action_approval(project, wired, command_calls, actor):
    actions, _, _ = wired
    set_safe_mode(False)

    pending = actions.dispatch(
        "run_command", {"cmd": "echo hello"}, actor=actor,
        offer_safe_mode_confirmation=True,
    )

    assert pending["needs_confirmation"]
    assert pending["requires_safe_mode_off"] is False
    assert "safe mode" not in pending["reason"].lower()
    assert not command_calls
    mode_changes = []
    result = actions.confirm(
        pending["token"], allow_safe_mode_change=False,
        on_safe_mode_change=lambda: mode_changes.append(True),
    )
    assert result["returncode"] == 0
    assert command_calls == [pending["args"]]
    assert not mode_changes and not is_safe_mode()


@pytest.mark.parametrize("actor", ["user", "model"])
def test_pending_permission_after_manual_off_runs_once_without_changing_mode(
    project, wired, command_calls, actor,
):
    actions, _, _ = wired
    set_safe_mode(True)
    pending = actions.dispatch(
        "run_command", {"cmd": 'echo "exact command"'}, actor=actor,
        offer_safe_mode_confirmation=True,
    )
    assert pending["requires_safe_mode_off"] is True

    set_safe_mode(False)
    assert not command_calls
    mode_changes = []
    result = actions.confirm(
        pending["token"], allow_safe_mode_change=False,
        on_safe_mode_change=lambda: mode_changes.append(True),
    )

    assert result["returncode"] == 0
    assert command_calls == [pending["args"]]
    assert not mode_changes and not is_safe_mode()
    assert "already used" in actions.confirm(pending["token"])["error"]
    assert command_calls == [pending["args"]]


@pytest.mark.parametrize("actor", ["user", "model"])
def test_action_only_approval_cannot_disable_mode_reenabled_after_manual_off(
    project, wired, command_calls, actor,
):
    actions, _, _ = wired
    set_safe_mode(True)
    pending = actions.dispatch(
        "run_command", {"cmd": "echo hello"}, actor=actor,
        offer_safe_mode_confirmation=True,
    )
    assert pending["requires_safe_mode_off"] is True
    set_safe_mode(False)
    # Another client changes mode after the user sees the ordinary run prompt.
    set_safe_mode(True)
    mode_changes = []

    result = actions.confirm(
        pending["token"], allow_safe_mode_change=False,
        on_safe_mode_change=lambda: mode_changes.append(True),
    )

    assert result["denied"]
    assert is_safe_mode() and not command_calls and not mode_changes
    assert "already used" in actions.confirm(pending["token"])["error"]


def test_hud_request_after_manual_toggle_off_has_no_mode_change_permission(hud, command_calls):
    set_safe_mode(True)
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        assert receive(ws, "status")["safe_mode"] is True
        ws.send_json({"type": "set_safe_mode", "enabled": False})
        assert receive(ws, "status")["safe_mode"] is False

        submit(ws, preview(ws, "gti status"))
        pending = receive(ws, "confirm_request")
        assert pending["requires_safe_mode_off"] is False
        assert "safe mode" not in pending["reason"].lower()
        assert not command_calls

        ws.send_json({"type": "confirm", "token": pending["token"], "granted": True})
        assert "completed" in receive(ws, "reply")["text"]
        assert receive(ws, "status")["safe_mode"] is False
        assert command_calls == [pending["args"]]


@pytest.mark.parametrize("source", ["direct", "model"])
def test_hud_action_only_approval_preserves_reenabled_mode(hud, command_calls, source):
    set_safe_mode(True)
    client, _ = hud

    class LLM:
        calls = 0
        observation = ""

        def chat(self, messages, on_token=None):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"tool": "run_command", "args": {"cmd": "echo hello"}})
            self.observation = messages[-1]["content"]
            return json.dumps({"reply": "Could not execute."})

    llm = LLM()
    server._state["brain"].llm = llm
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        if source == "direct":
            submit(ws, preview(ws, "gti status"))
        else:
            ws.send_json({"type": "input", "text": "please demonstrate a greeting"})
        pending = receive(ws, "confirm_request")
        assert pending["requires_safe_mode_off"] is True
        for enabled in (False, True):
            ws.send_json({"type": "set_safe_mode", "enabled": enabled})
            assert receive(ws, "status")["safe_mode"] is enabled

        ws.send_json({
            "type": "confirm", "token": pending["token"], "granted": True,
            "allow_safe_mode_change": False,
        })
        if source == "direct":
            assert "Safe Mode is ON" in receive(ws, "error")["text"]
        else:
            assert "Could not execute." in receive(ws, "reply")["text"]
            assert "Safe Mode is ON" in llm.observation
        assert is_safe_mode() and not command_calls


@pytest.mark.parametrize("allow_mode_change", ["false", "true", 0, 1, None, [], {}])
def test_hud_mode_change_consent_requires_boolean_without_consuming_token(
    hud, command_calls, allow_mode_change,
):
    set_safe_mode(True)
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        submit(ws, preview(ws, "gti status"))
        pending = receive(ws, "confirm_request")
        ws.send_json({
            "type": "confirm", "token": pending["token"], "granted": True,
            "allow_safe_mode_change": allow_mode_change,
        })
        assert "boolean" in receive(ws, "error")["text"]
        assert pending["token"] in server._confirmations
        assert is_safe_mode() and not command_calls
        ws.send_json({"type": "confirm", "token": pending["token"], "granted": False})
        assert "Cancelled" in receive(ws, "reply")["text"]
