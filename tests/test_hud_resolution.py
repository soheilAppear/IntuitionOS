"""Command review and capability enforcement through the real WebSocket."""

from pathlib import Path
from dataclasses import replace

import pytest
import yaml
from fastapi.testclient import TestClient

from core.capabilities import capabilities, set_safe_mode
from core.command_resolver import (
    CommandResolver,
    CorrectionFeedbackStore,
    IntuitionCommandProvider,
    StaticCommandProvider,
)
from interface import server


@pytest.fixture
def hud(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "config/config.yaml").read_text(encoding="utf-8"))
    cfg.update(
        memory_db_path=str(tmp_path / "test.db"),
        log_path=str(tmp_path / "test.log"),
        system_prompt_path=str(repo / "config/system_prompt.txt"),
        planner_schema_path=str(repo / "config/planner_schema.json"),
        voice={"enabled": False},
        hardware={"drivers": []},
        anticipation={"enabled": False},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "_load_config", lambda: cfg)
    with TestClient(server.app) as client:
        feedback = CorrectionFeedbackStore(
            server._state["mem"], enabled=lambda: server._state["episodes"].enabled
        )
        server._state["resolver"] = CommandResolver(
            [
                IntuitionCommandProvider(),
                StaticCommandProvider(["git", "python", "echo"], case_sensitive=False),
            ],
            shell="cmd",
            feedback=feedback,
        )
        yield client, cfg


def receive(ws, kind):
    for _ in range(12):
        message = ws.receive_json()
        if message["type"] == kind:
            return message
        if message["type"] == "error" and kind != "error":
            pytest.fail(str(message))
    pytest.fail(f"No {kind} message")


def preview(ws, text, revision=1):
    ws.send_json({"type": "buffer", "text": text, "client_revision": revision})
    return receive(ws, "resolution")


def submit(ws, snapshot, index=0, **overrides):
    message = {
        "type": "input",
        "text": snapshot["original"],
        "selected_text": snapshot["original"]
        if index is None
        else snapshot["candidates"][index]["text"],
        "token": snapshot["token"],
        "revision": snapshot["revision"],
        "candidate_index": index,
    }
    message.update(overrides)
    ws.send_json(message)


def test_hud_displays_raw_preserving_correction_and_safe_mode_blocks_execution(hud):
    set_safe_mode(True)
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, '  pyhton\ttrain.py --key "secret-value"  ')
        assert (
            shown["candidates"][0]["text"]
            == '  python\ttrain.py --key "secret-value"  '
        )
        assert shown["candidates"][0]["span"] == [2, 8]
        submit(ws, shown)
        assert "Safe Mode" in receive(ws, "reply")["text"]
        feedback = server._state["mem"].query(
            "SELECT original_token,selected_token,accepted,outcome,candidates_json FROM command_corrections"
        )
        assert feedback[0][:4] == ("pyhton", "python", 1, "denied")
        assert "secret-value" not in str(feedback)
        assert "secret-value" not in str(server._state["episodes"].all())
        assert server.get_journal().recent()[0]["decision"] == "deny"


def test_hud_binds_confirmed_execution_to_displayed_arguments_once(hud, monkeypatch):
    set_safe_mode(False)
    calls = []
    monkeypatch.setitem(
        capabilities._caps,
        "run_command",
        replace(
            capabilities.get("run_command"),
            fn=lambda **kwargs: (
                calls.append(kwargs)
                or {"stdout": "showcase: reviewed command", "returncode": 0}
            ),
        ),
    )
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, 'gti\tstatus  -- "file name"')
        submit(ws, shown)
        confirmation = receive(ws, "confirm_request")
        assert not calls
        assert confirmation["args"]["cmd"] == 'git\tstatus  -- "file name"'
        ws.send_json(
            {
                "type": "confirm",
                "token": confirmation["token"],
                "granted": True,
                "args": {"cmd": "malicious replacement"},
            }
        )
        assert "reviewed command" in receive(ws, "reply")["text"]
        receive(ws, "status")
        assert calls[0]["cmd"] == 'git\tstatus  -- "file name"'
        ws.send_json(
            {"type": "confirm", "token": confirmation["token"], "granted": True}
        )
        assert "expired" in receive(ws, "error")["text"].lower()
        assert len(calls) == 1


def test_edit_invalidates_both_displayed_correction_and_parked_approval(
    hud, monkeypatch
):
    set_safe_mode(False)
    calls = []
    monkeypatch.setitem(
        capabilities._caps,
        "run_command",
        replace(
            capabilities.get("run_command"),
            fn=lambda **kwargs: calls.append(kwargs) or {},
        ),
    )
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        stale = preview(ws, "gti status")
        fresh = preview(ws, "gti status --short", revision=2)
        submit(ws, stale)
        assert "Stale" in receive(ws, "error")["text"]
        submit(ws, fresh)
        confirmation = receive(ws, "confirm_request")
        preview(ws, "", revision=3)
        ws.send_json(
            {"type": "confirm", "token": confirmation["token"], "granted": True}
        )
        assert "expired" in receive(ws, "error")["text"].lower()
        assert not calls


def test_other_connection_cannot_submit_a_resolution_or_confirm_its_action(hud):
    set_safe_mode(False)
    client, _ = hud
    with (
        client.websocket_connect("/ws") as first,
        client.websocket_connect("/ws") as other,
    ):
        receive(first, "status")
        receive(other, "status")
        shown = preview(first, "gti status")
        submit(other, shown)
        assert "Stale" in receive(other, "error")["text"]
        submit(first, shown)
        confirmation = receive(first, "confirm_request")
        other.send_json(
            {"type": "confirm", "token": confirmation["token"], "granted": True}
        )
        assert "another connection" in receive(other, "error")["text"]
        first.send_json(
            {"type": "confirm", "token": confirmation["token"], "granted": False}
        )
        assert "Cancelled" in receive(first, "reply")["text"]


def test_no_silent_correction_for_legacy_submission_and_original_is_available(hud):
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        ws.send_json({"type": "input", "text": "/hlep"})
        shown = receive(ws, "resolution")
        assert shown["candidates"][0]["text"] == "/help"
        submit(ws, shown, index=None)
        assert "Unknown command: /hlep" in receive(ws, "error")["text"]
        fresh = preview(ws, "/hlep", revision=2)
        submit(ws, fresh)
        assert "Commands:" in receive(ws, "reply")["text"]


def test_argument_tampering_and_selection_replay_are_rejected(hud):
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, "/hlep")
        submit(ws, shown, selected_text="/safe off")
        receive(ws, "error")
        submit(ws, shown)
        receive(ws, "reply")
        submit(ws, shown)
        receive(ws, "error")


def test_git_original_runs_unchanged_only_through_gate(hud):
    set_safe_mode(False)
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, "git statsu")
        assert shown["candidates"][0]["text"] == "git status"
        submit(ws, shown, index=None)
        confirmation = receive(ws, "confirm_request")
        assert confirmation["args"]["cmd"] == "git statsu"


def test_manual_edit_feedback_and_forget_clear_every_connection(hud):
    client, _ = hud
    with (
        client.websocket_connect("/ws") as ws,
        client.websocket_connect("/ws") as other,
    ):
        receive(ws, "status")
        receive(other, "status")
        stale = preview(other, "/hlep")
        preview(ws, "gti status --secret abc")
        preview(ws, "git status --secret def", revision=2)
        rows = server._state["mem"].query(
            "SELECT manual_token,accepted FROM command_corrections WHERE original_token='gti'"
        )
        assert rows == [("git", None)]
        ws.send_json({"type": "input", "text": "/forget"})
        receive(ws, "reply")
        receive(other, "input_invalidated")
        assert (
            server._state["mem"].query("SELECT COUNT(*) FROM command_corrections")[0][0]
            == 0
        )
        assert server._state["episodes"].count() == 0
        submit(other, stale)
        receive(other, "error")


def test_logging_disable_drops_pending_views_and_stops_learning(hud):
    client, cfg = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        stale = preview(ws, "/hlep")
        cfg["episodes"] = {"enabled": False}
        ws.send_json({"type": "input", "text": "/reload"})
        receive(ws, "reply")
        before = server._state["mem"].query("SELECT COUNT(*) FROM command_corrections")[
            0
        ][0]
        submit(ws, stale)
        receive(ws, "error")
        fresh = preview(ws, "/hlep", revision=2)
        submit(ws, fresh)
        receive(ws, "reply")
        assert (
            server._state["mem"].query("SELECT COUNT(*) FROM command_corrections")[0][0]
            == before
        )


def test_advertised_hud_commands_have_handlers(hud):
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        for text in (
            "/help",
            "/config",
            '/write example.txt "hello review"',
            "/hw schema missing",
            '/task_payload \'{"action": "list_dir", "args": {"path": "."}}\' in 1h',
        ):
            ws.send_json({"type": "input", "text": text})
            reply = receive(ws, "reply")
            assert "Unknown command" not in reply["text"]
        assert Path("example.txt").read_text() == "hello review"


def test_voice_transcript_is_draft_and_never_executes(hud):
    client, _ = hud

    class FakeVoice:
        def is_recording(self):
            return False

        def is_busy(self):
            return False

        def stop_now(self):
            pass

        def start_recording_vad(self, on_silence, on_complete, on_error=None):
            on_complete("gti status")

    server._state["voice"] = FakeVoice()
    server._state["voice_status"] = {
        "state": "ready",
        "available": True,
        "text": "Voice is ready.",
    }
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        ws.send_json({"type": "voice_start"})
        assert receive(ws, "voice_text")["text"] == "gti status"
        ws.send_json({"type": "get_status"})
        receive(ws, "status")
        assert server._state["episodes"].count() == 0
        assert not server._confirmations


@pytest.mark.parametrize(
    "text,action",
    [
        ("set volume to 50", "os_set_volume"),
        ("open chrome", "os_open_app"),
        ("battery status", "os_get_battery"),
    ],
)
def test_existing_os_intentions_remain_exact_and_reach_their_capability(
    hud, monkeypatch, text, action
):
    calls = []
    monkeypatch.setitem(
        capabilities._caps,
        action,
        replace(
            capabilities.get(action),
            fn=lambda **kwargs: calls.append(kwargs) or {"ok": True},
        ),
    )
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, text)
        assert shown["status"] == "exact"
        assert shown["namespace"] == "intuitionos"
        assert shown["candidates"] == []
        submit(ws, shown, index=None)
        receive(ws, "reply")
        assert calls


def test_explicit_exec_retains_all_unquoted_arguments(hud):
    set_safe_mode(False)
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, '/exec\tgti\tstatus  -- "space name"  ')
        assert shown["candidates"][0]["text"] == '/exec\tgit\tstatus  -- "space name"  '
        submit(ws, shown)
        confirmation = receive(ws, "confirm_request")
        assert confirmation["args"]["cmd"] == 'git\tstatus  -- "space name"  '
        assert confirmation["capability"] == "run_command"


def test_supported_shell_failure_is_feedback_error_not_rejected_interpretation(
    hud, monkeypatch
):
    set_safe_mode(False)
    monkeypatch.setitem(
        capabilities._caps,
        "run_command",
        replace(
            capabilities.get("run_command"),
            fn=lambda **kwargs: {
                "stdout": "",
                "stderr": "no repository",
                "returncode": 128,
            },
        ),
    )
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, "gti status")
        submit(ws, shown)
        confirmation = receive(ws, "confirm_request")
        ws.send_json(
            {"type": "confirm", "token": confirmation["token"], "granted": True}
        )
        receive(ws, "reply")
        receive(ws, "status")
        rows = server._state["mem"].query(
            "SELECT accepted,outcome FROM command_corrections"
        )
        assert rows == [(1, "error")]


def test_unsupported_shell_operators_are_explained_without_execution(hud, monkeypatch):
    calls = []
    monkeypatch.setitem(
        capabilities._caps,
        "run_command",
        replace(
            capabilities.get("run_command"),
            fn=lambda **kwargs: calls.append(kwargs) or {},
        ),
    )
    client, _ = hud
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, "gti status | echo unsafe")
        assert shown["status"] == "unsupported"
        assert not shown["candidates"]
        submit(ws, shown, index=None)
        assert "safe correction span" in receive(ws, "error")["text"]
        assert not calls


def test_shutdown_cannot_restore_forgotten_predictor(tmp_path, monkeypatch):
    from core.episodes import Episode
    from core.predictor import Predictor, PredictorStore

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "config/config.yaml").read_text(encoding="utf-8"))
    cfg.update(
        memory_db_path=str(tmp_path / "test.db"),
        log_path=str(tmp_path / "test.log"),
        system_prompt_path=str(repo / "config/system_prompt.txt"),
        planner_schema_path=str(repo / "config/planner_schema.json"),
        voice={"enabled": False},
        hardware={"drivers": []},
        anticipation={"enabled": False},
    )
    monkeypatch.setattr(server, "_load_config", lambda: cfg)
    monkeypatch.chdir(tmp_path)
    with TestClient(server.app) as client:
        previous = server._state["predictor"]
        previous.update(Episode(ts=1, action="git status"))
        previous.save()
        with client.websocket_connect("/ws") as ws:
            receive(ws, "status")
            ws.send_json({"type": "input", "text": "  /forget  "})
            receive(ws, "reply")
        assert server._state["predictor"] is not previous
        assert server._state["predictor"].seen == 0
        assert server._state["episodes"].count() == 0
        mem = server._state["mem"]
    # Startup used to capture the old predictor in lifespan's local variable,
    # and save that forgotten model right here, when the server shut down.
    assert Predictor(store=PredictorStore(mem)).seen == 0
