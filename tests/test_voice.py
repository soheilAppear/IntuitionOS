"""Voice lifecycle without opening a real microphone or loading/downloading models."""

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.voice import VoiceRecognizer
from interface import server
from test_hud_resolution import hud as hud  # Re-export the shared pytest fixture.
from test_hud_resolution import receive


def fake_audio(monkeypatch, *, speech=True, fail=None, no_callbacks=False):
    streams = []

    class Stream:
        def __init__(self, **kwargs):
            if fail == "open":
                raise RuntimeError("selected microphone is unavailable")
            self.callback = kwargs["callback"]
            self.closed = False
            streams.append(self)

        def start(self):
            if fail == "start":
                raise RuntimeError("device start failed")
            if not no_callbacks:
                chunk = np.full((1024, 1), 0.1 if speech else 0, dtype=np.float32)
                self.callback(chunk, 1024, None, None)

        def stop(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=Stream))
    return streams


@pytest.mark.parametrize("phase", ["open", "start"])
def test_device_failure_reports_error_and_releases_capture(monkeypatch, phase):
    streams = fake_audio(monkeypatch, fail=phase)
    voice = VoiceRecognizer()
    errors, completed = [], []
    thread = voice.start_recording_vad(
        on_error=errors.append, on_complete=completed.append, max_sec=0.01
    )
    thread.join(1)
    assert not thread.is_alive()
    assert errors and errors[0].startswith("Microphone error:")
    assert not completed
    assert not voice.is_recording() and not voice.is_busy()
    assert all(stream.closed for stream in streams)


def test_missing_audio_dependency_is_visible(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    voice = VoiceRecognizer()
    errors = []
    thread = voice.start_recording_vad(on_error=errors.append)
    thread.join(1)
    assert errors and "sounddevice" in errors[0]
    assert not voice.is_busy()


def test_transcription_failure_after_mic_closes_always_reports_terminal_event(
    monkeypatch,
):
    streams = fake_audio(monkeypatch)
    voice = VoiceRecognizer()

    def fail(_frames):
        raise RuntimeError("Whisper model cannot be read")

    monkeypatch.setattr(voice, "_transcribe_frames", fail)
    events = []
    thread = voice.start_recording_vad(
        on_silence=lambda: events.append("closed"),
        on_error=lambda error: events.append(error),
        on_complete=lambda text: events.append("success"),
        max_sec=0.01,
    )
    thread.join(1)
    assert events == ["closed", "Transcription error: Whisper model cannot be read"]
    assert not voice.is_busy() and streams[0].closed


def test_device_without_audio_callbacks_still_finishes_on_wall_clock_deadline(
    monkeypatch,
):
    streams = fake_audio(monkeypatch, no_callbacks=True)
    voice = VoiceRecognizer()
    completed = []
    thread = voice.start_recording_vad(on_complete=completed.append, max_sec=0.01)
    thread.join(1)
    assert not thread.is_alive()
    assert completed == [""]
    assert streams[0].closed and not voice.is_recording()


def test_manual_stop_without_callbacks_finishes_and_parallel_start_is_rejected(
    monkeypatch,
):
    fake_audio(monkeypatch, no_callbacks=True)
    voice = VoiceRecognizer()
    completed = []
    thread = voice.start_recording_vad(on_complete=completed.append, max_sec=15)
    with pytest.raises(RuntimeError, match="already recording"):
        voice.start_recording_vad()
    voice.stop_now()
    thread.join(1)
    assert not thread.is_alive()
    assert completed == [""] and not voice.is_busy()


def test_disconnected_callback_cannot_leave_busy_flag_set(monkeypatch):
    fake_audio(monkeypatch, speech=False)
    voice = VoiceRecognizer()

    def disconnected(*args):
        raise RuntimeError("event loop closed")

    thread = voice.start_recording_vad(
        on_complete=disconnected, on_silence=disconnected, max_sec=0.01
    )
    thread.join(1)
    assert not thread.is_alive() and not voice.is_busy()


def test_model_preparation_loads_once_and_never_opens_audio(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(
            WhisperModel=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or object()
            )
        ),
    )
    voice = VoiceRecognizer()
    voice.prepare()
    voice.prepare()
    assert len(calls) == 1
    assert calls[0][1] == {"device": "cpu", "compute_type": "int8"}


class StubVoice:
    def __init__(self, *, outcome="error"):
        self.outcome = outcome
        self.started = False
        self.stopped = False

    def is_recording(self):
        return False

    def is_busy(self):
        return False

    def stop_now(self):
        self.stopped = True

    def start_recording_vad(self, on_silence, on_complete, on_error):
        self.started = True
        if self.outcome == "error":
            on_silence()
            on_error("Transcription error: model failed")
        elif self.outcome == "device":
            on_error("Microphone error: device unavailable")
        else:
            on_silence()
            on_complete("")


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("error", "Transcription error"),
        ("device", "Microphone error"),
        ("empty", "No speech detected"),
    ],
)
def test_voice_failures_reach_socket_and_leave_retry_available(hud, outcome, expected):
    client, _ = hud
    voice = StubVoice(outcome=outcome)
    server._state.update(
        voice=voice, voice_status={"state": "ready", "available": True, "text": "Ready"}
    )
    with client.websocket_connect("/ws") as ws:
        assert receive(ws, "status")["voice"]["available"]
        ws.send_json({"type": "voice_start"})
        error = receive(ws, "error")
        assert error["source"] == "voice" and expected in error["text"]
        ws.send_json({"type": "get_status"})
        status = receive(ws, "status")["voice"]
        assert status["state"] in ("ready", "error")
        assert status["available"] is True
        assert voice.started
        assert server._state["episodes"].count() == 0


@pytest.mark.parametrize("state", ["disabled", "loading", "error"])
def test_voice_setup_status_blocks_recording_with_a_specific_message(hud, state):
    client, _ = hud
    voice = StubVoice()
    text = f"Voice {state}: setup detail"
    server._state.update(
        voice=voice, voice_status={"state": state, "available": False, "text": text}
    )
    with client.websocket_connect("/ws") as ws:
        assert receive(ws, "status")["voice"]["state"] == state
        ws.send_json({"type": "voice_start"})
        error = receive(ws, "error")
        assert error == {"type": "error", "source": "voice", "text": text}
        assert not voice.started


@pytest.mark.parametrize("failure", [False, True])
def test_startup_preloader_reports_ready_or_failure_without_recording(
    tmp_path, monkeypatch, failure
):
    from pathlib import Path
    import yaml
    from fastapi.testclient import TestClient

    ready = threading.Event()

    class PreparedVoice(StubVoice):
        def __init__(self, model_size, language):
            super().__init__()
            self.model_size = model_size

        def prepare(self):
            ready.set()
            if failure:
                raise RuntimeError("local model files are missing")

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "config/config.yaml").read_text(encoding="utf-8"))
    cfg.update(
        memory_db_path=str(tmp_path / "test.db"),
        log_path=str(tmp_path / "test.log"),
        system_prompt_path=str(repo / "config/system_prompt.txt"),
        planner_schema_path=str(repo / "config/planner_schema.json"),
        voice={"enabled": True, "model": "base"},
        hardware={"drivers": []},
        anticipation={"enabled": False},
    )
    monkeypatch.setattr(server, "_load_config", lambda: cfg)
    monkeypatch.setattr(server, "VoiceRecognizer", PreparedVoice)
    monkeypatch.chdir(tmp_path)
    with TestClient(server.app) as client:
        assert ready.wait(1)
        with client.websocket_connect("/ws") as ws:
            status = receive(ws, "status")["voice"]
            for _ in range(10):
                if status["state"] != "loading":
                    break
                ws.send_json({"type": "get_status"})
                status = receive(ws, "status")["voice"]
            assert status["state"] == ("error" if failure else "ready")
            assert status["available"] is not failure
            if failure:
                assert "local model files are missing" in status["text"]
            assert not server._state["voice"].started
