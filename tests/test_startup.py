"""Boot both interfaces for real.

Most of this project's wiring lives in two startup functions that no unit test
touches, so a misordered assignment or a renamed return value would otherwise
only show up when somebody launched the app. These tests are cheap insurance:
they build the whole object graph, exercise a websocket round trip, and shut it
down again.
"""

import asyncio
import json
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
    for key in (
        "cfg",
        "brain",
        "mem",
        "sched",
        "ant",
        "episodes",
        "sensor",
        "predictor",
        "calibrator",
        "calibration_store",
        "thresholds",
        "rules",
    ):
        assert state.get(key) is not None, f"startup did not provide {key!r}"


@pytest.mark.parametrize("abort", [False, True])
def test_brainbit_stays_idle_at_startup_and_closes_on_lifespan_exit(app_dir, monkeypatch, abort):
    from interface import server
    from plugins import brainbit

    path = app_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["hardware"] = {"drivers": [{"name": "brainbit", "enabled": True}]}
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(brainbit.importlib.util, "find_spec", lambda name: object())

    def unexpected_worker(*args, **kwargs):
        pytest.fail("Booting the app must not open a native BrainBit worker")

    monkeypatch.setattr(brainbit, "_WorkerClient", unexpected_worker)

    async def boot():
        try:
            async with server.lifespan(server.app):
                driver = server._state["brainbit"]
                assert driver.status()["state"] == "disconnected"
                assert driver._worker is None
                if abort:
                    raise RuntimeError("test lifespan cancellation")
        except RuntimeError:
            assert abort
        finally:
            # Older unrelated services do not have exceptional-exit cleanup.
            server._state["ant"].stop()
            server._state["sched"].stop()
        assert driver.status()["state"] == "disabled"

    asyncio.run(boot())


def test_startup_registers_the_os_capabilities_with_a_declared_cost(app_dir):
    from core.capabilities import capabilities
    from interface import server

    async def boot():
        async with server.lifespan(server.app):
            return sorted(capabilities.names())

    names = asyncio.run(boot())
    assert "os_shutdown_computer" in names
    assert capabilities.get("os_shutdown_computer").requires_confirmation


@pytest.mark.parametrize("enabled", [None, False, True])
def test_hand_bend_click_config_reaches_idle_recognizer_at_startup(app_dir, enabled):
    from interface import server

    path = app_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if enabled is None:
        cfg["gestures"].pop("bend_click", None)
    else:
        cfg["gestures"]["bend_click"] = enabled
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    async def boot():
        async with server.lifespan(server.app):
            recognizer = server._state["gestures"]
            assert recognizer.bend_click is (enabled is True)
            assert server._state["gesture_settings"]["bend_click"] is (enabled is True)
            assert recognizer._mouse is None
            assert recognizer.is_running() is False

    asyncio.run(boot())


def test_mouse_mode_authorizes_navigation_only_during_a_live_clutch(app_dir, monkeypatch):
    from interface import server

    async def boot():
        async with server.lifespan(server.app):
            recognizer = server._state["gestures"]
            controls = server._state["hand_controls"]
            recognizer.input_mode = "mouse"
            monkeypatch.setattr(recognizer, "is_running", lambda: True)
            assert not controls.active()
            assert "error" in controls._authorize_desktop()
            recognizer._navigation_active = True
            assert controls.active()
            assert controls._authorize_desktop()["ok"]
            assert controls._authorize_overview()["ok"]
            monkeypatch.setattr(recognizer, "is_running", lambda: False)
            assert not controls.active()
            assert "error" in controls._authorize_overview()
            recognizer._navigation_active = False

    asyncio.run(boot())


@pytest.mark.parametrize("travel", [None, 2.4])
def test_navigation_travel_default_and_user_setting_reach_both_modes(app_dir, travel):
    from interface import server

    path = app_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if travel is None:
        cfg["gestures"].pop("travel_palms", None)
    else:
        cfg["gestures"]["travel_palms"] = travel
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    expected = 1.2 if travel is None else travel

    async def boot():
        async with server.lifespan(server.app):
            recognizer = server._state["gestures"]
            assert recognizer.measured.travel_palms == expected
            assert recognizer._navigation.travel_palms == expected
            assert server._state["gesture_settings"]["travel_palms"] == expected

    asyncio.run(boot())


def test_the_predictor_is_wired_to_the_rules_and_the_calibrator(app_dir):
    from interface import server

    async def boot():
        async with server.lifespan(server.app):
            return server._state["predictor"]

    predictor = asyncio.run(boot())
    assert predictor.rules is not None, "consolidation rules never reach the predictor"
    assert predictor.calibrator is not None, (
        "the calibrator never reaches the predictor"
    )


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

    cfg, brain, mem, sched, episodes, sensor, predictor, rules, calib = (
        terminal.bootstrap()
    )
    try:
        assert cfg and brain and mem and episodes and sensor and predictor
        assert rules.all() == []
        assert predictor.rules is rules
    finally:
        sched.stop()


def test_terminal_startup_wires_reminders_to_the_shared_memory(app_dir):
    """Memory is created before the scheduler; binding must work in that order."""
    from core.actions import actions
    from interface import terminal

    _, _, mem, sched, *_ = terminal.bootstrap()
    try:
        result = actions.call(
            "create_task", text="verify terminal wiring", when="in 10m"
        )
        assert result.get("ok"), result
        assert sched.memory is mem
        assert mem.get_task(result["id"])["title"] == "verify terminal wiring"
    finally:
        sched.stop()


def test_fresh_terminal_registers_and_runs_web_lookup(app_dir, monkeypatch):
    """No prior HUD session may be needed to make web tools available."""
    from core import actions as actions_mod, os_sandbox
    from core.capabilities import capabilities
    from interface import terminal

    # Earlier HUD tests can hide missing startup registration. Use fresh
    # registry dictionaries and restore the original ones when this test ends.
    monkeypatch.setattr(capabilities, "_caps", {
        name: cap for name, cap in capabilities._caps.items()
        if not name.startswith("os_")
    })
    monkeypatch.setattr(actions_mod.actions, "_actions", {
        name: fn for name, fn in actions_mod.actions._actions.items()
        if not name.startswith("os_")
    })
    monkeypatch.setattr(actions_mod.actions, "names", {
        name for name in actions_mod.actions.names if not name.startswith("os_")
    })
    fetched = []

    def fetch(url, max_chars=4000):
        fetched.append(url)
        return {"ok": True, "url": url, "text": "Boston: Cloudy +18 C"}

    monkeypatch.setattr(os_sandbox, "fetch_url", fetch)
    _, brain, mem, sched, *_ = terminal.bootstrap()
    try:
        assert "os_fetch_url(" in brain.build_system_prompt()
        assert capabilities.get("os_shutdown_computer").requires_confirmation
        outputs = iter([
            json.dumps({"tool": "os_fetch_url", "args": {"url": "https://example.com/weather"}}),
            json.dumps({"reply": "Boston is cloudy, 18 C."}),
        ])
        monkeypatch.setattr(brain.llm, "chat_json", lambda messages, on_token=None: next(outputs))
        result = brain.step("Tell me the weather in Boston")
        assert fetched == ["https://example.com/weather"]
        assert result["reply"] == "Boston is cloudy, 18 C."
        assert "error" not in result
    finally:
        sched.stop()
        mem.close()


def test_hud_startup_can_create_a_reminder_without_a_prior_session(app_dir):
    """Exercise fresh startup through the same socket command a person uses."""
    from fastapi.testclient import TestClient
    from interface import server

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json(
                {"type": "input", "text": "remind me verify HUD wiring in 10m"}
            )
            reply = ws.receive_json()
            assert reply["type"] == "reply"
            assert "Reminder set" in reply["text"], reply
            assert server._state["sched"].memory is server._state["mem"]
            assert server._state["mem"].list_open()[0]["title"] == "verify HUD wiring"
