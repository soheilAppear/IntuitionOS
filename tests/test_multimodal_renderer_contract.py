"""Keep the renderer's shared JSON fixture aligned with real service metadata."""
import json
from pathlib import Path

from core.multimodal import MultimodalPreview


class NoHardware:
    """Construction and status must never acquire a camera, headset, or mouse."""

    eeg_compatible = True

    def __getattr__(self, name):
        raise AssertionError("Hardware access attempted: " + name)


def contract_states():
    device = NoHardware()
    # Explicit options cover both supported acquisition paths without starting either.
    return [MultimodalPreview(device, device, pointer_controller=device,
                              camera_mouse_enabled=enabled, clock=lambda: 0).status()
            for enabled in (True, False)]


def test_renderer_fixture_matches_real_stopped_service():
    fixture = Path(__file__).parent / "fixtures" / "multimodal-service-status.json"
    assert json.loads(fixture.read_text(encoding="utf-8")) == contract_states()
