"""Fast CNN API and explicit archives cannot inherit five-second models."""

import pytest

from interface import server
from test_multimodal_protocol import HEADERS, api


def other_archives_forbidden():
    for name in ("load_latest_decoder", "save_archive", "load_latest_blink_decoder", "save_blink_archive",
                 "load_latest_calibration", "save_calibration_archive", "load_latest_cnn_decoder", "save_cnn_archive"):
        assert not hasattr(server, name), "Retired archive loader remains reachable"


def acknowledgments():
    saved = {"eeg_saved_model_file": "spectral.json", "eeg_saved_blink_model_file": "blink.json",
             "eeg_saved_calibration_file": "draft.json", "eeg_saved_cnn_model_file": "five-second.json",
             "eeg_saved_fast_cnn_model_file": "fast-cnn-blink-one-second.json"}
    server._state.update(saved)
    return saved
def test_fast_save_uses_only_separate_directory_prefix_and_acknowledgment(api, monkeypatch, tmp_path):
    preview, client = api
    saved = acknowledgments()
    other_archives_forbidden()
    folder, writes = tmp_path / "eeg-fast-cnn-blink-models", []
    archive = {"format": "intuitionos.eeg-fast-cnn-blink-model", "fake": True}
    monkeypatch.setattr(preview, "eeg_export_cnn_model", lambda: archive, raising=False)
    monkeypatch.setattr(server, "_EEG_FAST_CNN_MODEL_DIRECTORY", folder)
    monkeypatch.setattr(server, "save_fast_cnn_archive", lambda state, path: writes.append((state, path)) or path)
    response = client.post("/multimodal/eeg_save", headers=HEADERS, json={})
    assert response.status_code == 200
    state, path = writes[0]
    assert state == archive and path.parent == folder and path.name.startswith("fast-cnn-blink-")
    assert response.json()["eeg_control"]["saved_model_file"] == path.name
    assert str(folder) not in response.text and '"fake"' not in response.text
    assert all(server._state[key] == value for key, value in saved.items() if key != "eeg_saved_fast_cnn_model_file")
    assert preview.calls == [] and not response.json()["eeg_control"]["armed"]


def test_fast_load_uses_only_its_own_directory_and_no_validation_override(api, monkeypatch, tmp_path):
    preview, client = api
    saved = acknowledgments()
    other_archives_forbidden()
    folder, reads, restored = tmp_path / "eeg-fast-cnn-blink-models", [], object()

    def select(decoder, *, source):
        preview.calls.append(("eeg_use_saved_cnn_model", decoder, source))
        return {**preview.status(), "eeg_control": {"method": "cnn_blink_fast", "armed": False, "saved_model_source": source}}

    monkeypatch.setattr(preview, "eeg_use_saved_cnn_model", select, raising=False)
    monkeypatch.setattr(server, "_EEG_FAST_CNN_MODEL_DIRECTORY", folder)
    monkeypatch.setattr(server, "load_latest_fast_cnn_decoder", lambda path: (reads.append(path) or restored, folder / "fast-cnn-blink-saved.json"))
    response = client.post("/multimodal/eeg_load_saved", headers=HEADERS, json={})
    assert response.status_code == 200 and reads == [folder]
    assert preview.calls == [("eeg_use_saved_cnn_model", restored, "fast-cnn-blink-saved.json")]
    assert response.json()["eeg_control"]["saved_model_source"] == "fast-cnn-blink-saved.json"
    assert "eeg_saved_fast_cnn_model_file" not in server._state and str(folder) not in response.text
    assert all(server._state[key] == value for key, value in saved.items() if key != "eeg_saved_fast_cnn_model_file")




@pytest.mark.parametrize("operation", ["eeg_save", "eeg_load_saved"])
def test_fast_archive_failure_never_falls_back_to_a_valid_old_model(api, monkeypatch, operation):
    preview, client = api
    other_archives_forbidden()

    def failure(*_args):
        raise ValueError("No valid fast CNN archive")

    monkeypatch.setattr(server, "load_latest_fast_cnn_decoder", failure)
    monkeypatch.setattr(preview, "eeg_export_cnn_model", failure, raising=False)
    response = client.post("/multimodal/" + operation, headers=HEADERS, json={})
    assert response.status_code == 409 and "No valid fast CNN archive" in response.json()["error"]
    assert preview.calls == [] and not response.json()["eeg_control"]["armed"]


@pytest.mark.parametrize("operation", ["eeg_save_progress", "eeg_load_progress"])
def test_fast_has_no_cross_profile_draft_persistence(api, monkeypatch, operation):
    preview, client = api
    other_archives_forbidden()
    response = client.post("/multimodal/" + operation, headers=HEADERS, json={})
    assert response.status_code == 404 and preview.calls == []


@pytest.mark.parametrize("operation", ["eeg_save", "eeg_load_saved"])
@pytest.mark.parametrize("payload", [{"path": "../five-second.json"}, {"enabled": True}, {"use_unvalidated": True}])
def test_fast_archive_api_rejects_paths_implicit_arming_and_validation_override(api, operation, payload):
    preview, client = api
    response = client.post("/multimodal/" + operation, headers=HEADERS, json=payload)
    assert response.status_code == 400 and preview.calls == []


@pytest.mark.parametrize("operation", ["eeg_save", "eeg_load_saved"])
def test_fast_archives_are_private_to_local_trusted_requests(api, monkeypatch, operation):
    preview, client = api
    monkeypatch.setattr(server, "load_latest_fast_cnn_decoder", lambda *_: pytest.fail("Untrusted archive read"))
    monkeypatch.setattr(server, "save_fast_cnn_archive", lambda *_: pytest.fail("Untrusted archive write"))
    response = client.post("/multimodal/" + operation, headers={**HEADERS, "origin": "null"}, json={})
    assert response.status_code == 403 and preview.calls == []


@pytest.mark.parametrize("route", ["status", "preview"])
def test_only_fast_saved_metadata_is_published(api, route):
    _, client = api
    saved = acknowledgments()
    response = client.get("/multimodal/" + route, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["eeg_control"]["saved_model_file"] == saved["eeg_saved_fast_cnn_model_file"]
    assert "five-second.json" not in response.text and "draft.json" not in response.text


@pytest.mark.parametrize("operation", ["control_mode", "eeg_method", "eeg_target", "contact",
                                      "arm", "calibrate", "reset_calibration"])
def test_retired_options_have_no_api_backdoor(api, operation):
    preview, client = api
    response = client.post("/multimodal/" + operation, headers=HEADERS, json={})
    assert response.status_code == 404 and preview.calls == []
