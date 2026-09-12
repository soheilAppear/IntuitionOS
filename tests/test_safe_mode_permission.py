"""Explicit HUD permission grants change mode only for a valid owned token."""

from dataclasses import replace
import json
import threading

import pytest

from core.capabilities import capabilities, is_safe_mode, pending_confirmations, set_safe_mode
from interface import server
from test_hud_resolution import hud as hud  # Shared real WebSocket fixture.
from test_hud_resolution import preview, receive, submit


@pytest.fixture
def shell_calls(monkeypatch):
    calls = []
    monkeypatch.setitem(capabilities._caps, "run_command", replace(
        capabilities.get("run_command"),
        fn=lambda **args: calls.append(args) or {"returncode": 0, "stdout": "ran exact command"},
    ))
    return calls


def test_toggle_broadcasts_to_every_socket_without_changing_draft_or_granting(hud, shell_calls):
    set_safe_mode(True)
    client, _ = hud
    with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as other:
        receive(first, "status")
        receive(other, "status")
        shown = preview(first, "gti status")
        submit(first, shown)
        pending = receive(first, "confirm_request")
        revision = server._confirmations[pending["token"]]["revision"]
        other.send_json({"type": "set_safe_mode", "enabled": False})
        assert receive(other, "status")["safe_mode"] is False
        assert receive(first, "status")["safe_mode"] is False
        assert not shell_calls
        assert server._confirmations[pending["token"]]["revision"] == revision
        assert any(meta["text"] == "gti status" for meta in server._connections.values())
        other.send_json({"type": "set_safe_mode", "enabled": True})
        assert receive(other, "status")["safe_mode"] is True
        assert receive(first, "status")["safe_mode"] is True
        first.send_json({"type": "confirm", "token": pending["token"], "granted": False})
        assert "Cancelled" in receive(first, "reply")["text"]
        assert is_safe_mode()
        assert not shell_calls


@pytest.mark.parametrize("enabled", ["false", 0, 1, None, [], {}])
def test_toggle_requires_an_actual_boolean(hud, enabled):
    set_safe_mode(True)
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        ws.send_json({"type": "set_safe_mode", "enabled": enabled})
        assert receive(ws, "error")["source"] == "safe_mode"
        assert is_safe_mode()


def test_safe_permission_grant_updates_status_before_execution_and_is_single_use(hud, monkeypatch):
    set_safe_mode(True)
    executing = threading.Event()
    release = threading.Event()
    calls = []

    def run(**args):
        calls.append(args)
        executing.set()
        assert release.wait(5), "test must release the blocked action"
        return {"returncode": 0, "stdout": "finished"}

    monkeypatch.setitem(capabilities._caps, "run_command", replace(
        capabilities.get("run_command"), fn=run,
    ))
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        submit(ws, preview(ws, 'gti status -- "two words"'))
        pending = receive(ws, "confirm_request")
        assert pending["requires_safe_mode_off"] is True
        assert is_safe_mode() and not calls
        ws.send_json({
            "type": "confirm", "token": pending["token"], "granted": True,
            "args": {"cmd": "replacement"},
        })
        try:
            assert receive(ws, "status")["safe_mode"] is False
            assert executing.wait(2)
        finally:
            release.set()
        assert "finished" in receive(ws, "reply")["text"]
        receive(ws, "status")
        assert calls == [pending["args"]]
        ws.send_json({"type": "confirm", "token": pending["token"], "granted": True})
        assert "expired" in receive(ws, "error")["text"].lower()
        assert len(calls) == 1


@pytest.mark.parametrize("granted", ["false", "true", 1, None, [], {}])
def test_non_boolean_approval_never_changes_mode_or_consumes_token(hud, shell_calls, granted):
    set_safe_mode(True)
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        submit(ws, preview(ws, "gti status"))
        pending = receive(ws, "confirm_request")
        ws.send_json({"type": "confirm", "token": pending["token"], "granted": granted})
        assert "boolean" in receive(ws, "error")["text"]
        assert is_safe_mode() and not shell_calls
        assert pending["token"] in server._confirmations
        ws.send_json({"type": "confirm", "token": pending["token"], "granted": False})
        receive(ws, "reply")


def test_other_socket_and_edited_permission_cannot_disable_safe_mode(hud, shell_calls):
    set_safe_mode(True)
    client, _ = hud
    with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as other:
        receive(first, "status")
        receive(other, "status")
        submit(first, preview(first, "gti status"))
        pending = receive(first, "confirm_request")
        other.send_json({"type": "confirm", "token": pending["token"], "granted": True})
        assert "another connection" in receive(other, "error")["text"]
        assert is_safe_mode() and not shell_calls
        preview(first, "", revision=2)
        first.send_json({"type": "confirm", "token": pending["token"], "granted": True})
        assert "expired" in receive(first, "error")["text"].lower()
        assert is_safe_mode() and not shell_calls


def test_expired_permission_does_not_change_safe_mode(project, wired, shell_calls, monkeypatch):
    actions, _, _ = wired
    set_safe_mode(True)
    pending = actions.dispatch("run_command", {"cmd": "echo hi"}, offer_safe_mode_confirmation=True)
    monkeypatch.setattr(pending_confirmations, "ttl_s", -1)
    assert "expired" in actions.confirm(pending["token"])["error"]
    assert is_safe_mode() and not shell_calls


@pytest.mark.parametrize("actor,args", [
    ("scheduler", {"cmd": "echo hi"}),
    ("anticipator", {"cmd": "echo hi"}),
    ("user", {"cmd": 1}),
    ("user", {"cmd": "echo hi", "cwd": ".."}),
])
def test_other_gate_denials_never_offer_safe_mode_permission(project, wired, shell_calls, actor, args):
    actions, _, _ = wired
    set_safe_mode(True)
    result = actions.dispatch("run_command", args, actor=actor, offer_safe_mode_confirmation=True)
    assert result["denied"]
    assert not result.get("needs_confirmation")
    assert is_safe_mode() and not shell_calls


def test_new_validation_denial_on_approval_leaves_safe_mode_on(project, wired, shell_calls, monkeypatch):
    actions, _, _ = wired
    set_safe_mode(True)
    pending = actions.dispatch("run_command", {"cmd": "echo hi"}, offer_safe_mode_confirmation=True)
    monkeypatch.setitem(capabilities._caps, "run_command", replace(
        capabilities.get("run_command"), extra_validate=lambda args: "permission revoked",
    ))
    assert actions.confirm(pending["token"])["denied"]
    assert is_safe_mode() and not shell_calls


@pytest.mark.parametrize("granted", [True, False])
def test_brain_tool_permission_is_displayed_and_resumes_after_explicit_answer(hud, shell_calls, granted):
    set_safe_mode(True)
    client, _ = hud

    class LLM:
        calls = 0

        def chat(self, messages, on_token=None):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"tool": "run_command", "args": {"cmd": "echo hello"}})
            return json.dumps({"reply": "Action completed." if granted else "Cancelled."})

    llm = LLM()
    server._state["brain"].llm = llm
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        ws.send_json({"type": "input", "text": "please demonstrate a greeting"})
        pending = receive(ws, "confirm_request")
        assert pending["requires_safe_mode_off"] is True
        assert pending["args"]["cmd"] == "echo hello"
        assert llm.calls == 1 and not shell_calls and is_safe_mode()
        ws.send_json({"type": "confirm", "token": pending["token"], "granted": granted})
        if granted:
            assert receive(ws, "status")["safe_mode"] is False
        receive(ws, "reply")
        assert is_safe_mode() is not granted
        assert bool(shell_calls) is granted
        assert llm.calls == 2
