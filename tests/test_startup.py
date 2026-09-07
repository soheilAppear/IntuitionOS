"""Boot both interfaces for real.

Most of this project's wiring lives in two startup functions that no unit test
touches, so a misordered assignment or a renamed return value would otherwise
only show up when somebody launched the app. These tests are cheap insurance:
they build the whole object graph, exercise a websocket round trip, and shut it
down again.
"""

import asyncio
import shutil

import pytest
import yaml


@pytest.fixture
def app_dir(tmp_path, monkeypatch):
    """A working directory that looks enough like the repo to boot in."""
    root = tmp_path / "app"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Voice pulls a Whisper model; hardware drivers poke at real devices. Neither
    # is what this test is checking.
    cfg["voice"] = {"enabled": False}
    cfg["hardware"] = {"drivers": []}
    cfg["memory_db_path"] = str(root / "data" / "test.db")
    cfg["log_path"] = str(root / "data" / "log.txt")
    cfg["system_prompt_path"] = "config/system_prompt.txt"
    cfg["planner_schema_path"] = "config/planner_schema.json"

    with open(root / "config" / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    shutil.copy("config/system_prompt.txt", root / "config" / "system_prompt.txt")
    shutil.copy("config/planner_schema.json", root / "config" / "planner_schema.json")

    monkeypatch.chdir(root)
    return root


def test_the_server_starts_and_stops_cleanly(app_dir):
    from interface import server

    async def boot():
        async with server.lifespan(server.app):
            return dict(server._state)

    state = asyncio.run(boot())

    # Every collaborator the request handlers reach for must actually be there.
    for key in ("cfg", "brain", "mem", "sched", "ant", "episodes", "sensor",
                "predictor", "calibrator", "calibration_store", "thresholds", "rules"):
        assert state.get(key) is not None, f"startup did not provide {key!r}"


def test_startup_registers_the_os_capabilities_with_a_declared_cost(app_dir):
    from core.capabilities import capabilities
    from interface import server

    async def boot():
        async with server.lifespan(server.app):
            return sorted(capabilities.names())

    names = asyncio.run(boot())
    assert "os_shutdown_computer" in names
    assert capabilities.get("os_shutdown_computer").requires_confirmation


def test_the_predictor_is_wired_to_the_rules_and_the_calibrator(app_dir):
    from interface import server

    async def boot():
        async with server.lifespan(server.app):
            return server._state["predictor"]

    predictor = asyncio.run(boot())
    assert predictor.rules is not None, "consolidation rules never reach the predictor"
    assert predictor.calibrator is not None, "the calibrator never reaches the predictor"


def test_a_websocket_round_trip_works(app_dir):
    """The handshake the HUD performs on connect, end to end."""
    from fastapi.testclient import TestClient

    from interface import server

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            status = ws.receive_json()
            assert status["type"] == "status"
            assert "safe_mode" in status

            ws.send_json({"type": "input", "text": "/capabilities"})
            reply = ws.receive_json()
            assert reply["type"] == "reply"
            assert "write_file" in reply["text"]


def test_an_input_is_recorded_as_an_episode_over_the_socket(app_dir):
    """Involuntary encoding, verified through the real transport rather than by
    calling the log directly."""
    from fastapi.testclient import TestClient

    from interface import server

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "buffer", "text": "/capa"})
            ws.send_json({"type": "input", "text": "/capabilities"})
            ws.receive_json()

        episodes = server._state["episodes"]
        actions_logged = [e.action for e in episodes.recent()]
        assert "/capabilities" in actions_logged


def test_dream_and_rules_survive_an_empty_log(app_dir):
    """/dream used to be a stub. Now it runs for real, and the first thing a new
    user does is run it before they have any history."""
    from fastapi.testclient import TestClient

    from interface import server

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()

            ws.send_json({"type": "input", "text": "/rules"})
            assert "No rules yet" in ws.receive_json()["text"]

            ws.send_json({"type": "input", "text": "/dream"})
            msg = ws.receive_json()
            while msg["type"] == "thinking":
                msg = ws.receive_json()
            assert msg["type"] == "reply"
            assert msg["text"]


def test_calibration_and_journal_commands_answer_on_a_fresh_install(app_dir):
    from fastapi.testclient import TestClient

    from interface import server

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            for command, expected in (
                ("/calibration", "nothing to calibrate"),
                ("/journal", "journal is empty"),
                ("/episodes", None),
                ("/thresholds", "auto_execute"),
            ):
                ws.send_json({"type": "input", "text": command})
                reply = ws.receive_json()
                assert reply["type"] == "reply", f"{command} -> {reply}"
                if expected:
                    assert expected.lower() in reply["text"].lower()


def test_the_terminal_bootstraps(app_dir):
    """bootstrap() returns a widening tuple that several call sites unpack, so a
    forgotten call site is a real risk."""
    from interface import terminal

    cfg, brain, mem, sched, episodes, sensor, predictor, rules, calib = terminal.bootstrap()
    try:
        assert cfg and brain and mem and episodes and sensor and predictor
        assert rules.all() == []
        assert predictor.rules is rules
    finally:
        sched.stop()
