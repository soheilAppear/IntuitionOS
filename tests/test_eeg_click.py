"""EEG click tests use fake devices only; they never move or click the real mouse."""

import threading  # Exercise cleanup serialization with a deliberately blocked fake press.

import pytest  # Parameterize native failure cases without touching hardware.

from core.eeg_click import EegClickController  # Importing the actuator does not create its device.


class FakeMouse:
    """Record only button events; any attempted pointer access is a test failure."""

    def __init__(self):
        self.events = []  # Preserve attempted events, including native calls that report failure.
        self.block = None  # Simulate an existing physical button or keyboard modifier.
        self.failures = {}  # Map a button event to the number of failures still to inject.

    def blocked(self):
        return self.block  # The actuator must check physical input before claiming ownership.

    def _emit(self, event):
        self.events.append(event)  # A rejected operation can still have affected native input.
        if self.failures.get(event, 0):  # Fail only the configured number of attempted events.
            self.failures[event] -= 1  # Allow a later cleanup attempt to demonstrate recovery.
            raise OSError(event + " rejected")  # Match the real adapter's exception contract.

    def left_down(self):
        self._emit("down")  # A click may contain exactly one attempted press.

    def left_up(self):
        self._emit("up")  # Cleanup is allowed to retry release without repeating the press.

    def position(self):
        pytest.fail("EEG clicking must not inspect or choose a pointer target")  # Click the current pointer only.

    def move(self, *args):
        pytest.fail("EEG clicking must not move the pointer")  # No coordinate-based actuation is permitted.


def test_construction_reset_and_close_never_initialize_or_inject():
    calls = []  # Detect native construction even when no button event would occur.
    controller = EegClickController(device_factory=lambda: calls.append("created"))  # Remain entirely lazy.
    assert controller.reset() == {"ok": True, "cleanup_pending": False}  # Unused reset requires no device.
    assert controller.close() == {"ok": True, "cleanup_pending": False}  # Unused close also requires no device.
    assert calls == []  # Neither lifecycle operation initialized native input.


def test_success_is_one_down_up_at_current_pointer_and_ids_survive_reset():
    device = FakeMouse()  # The fake fails immediately if pointer movement or position is requested.
    controller = EegClickController(device_factory=lambda: device)  # Never construct the Win32 adapter.
    assert controller.click() == {"ok": True, "click_id": 1, "cleanup_pending": False}  # Complete one click.
    assert device.events == ["down", "up"]  # A successful click never leaves the button held.
    assert controller.close()["ok"]  # Closing a released controller adds no synthetic input.
    assert controller.click()["click_id"] == 2  # Explicit later authorization can reuse the actuator.
    assert device.events == ["down", "up", "down", "up"]  # Each call produces exactly one full click.


@pytest.mark.parametrize("reason", ["Release physical mouse buttons to use Hand Mouse",
                                  "Release keyboard modifier keys to use Hand Mouse"])
def test_physical_input_blocks_click_without_releasing_unowned_buttons(reason):
    device = FakeMouse()  # Represent a held physical input belonging to the user.
    device.block = reason  # Test both button and modifier conflicts.
    controller = EegClickController(device_factory=lambda: device)  # Use fake input only.
    result = controller.click()  # Refuse the click before attempting a synthetic press.
    assert not result["ok"] and "EEG clicking" in result["error"]  # Report the physical conflict clearly.
    assert result["cleanup_pending"] is False  # No synthetic button was acquired.
    assert controller.close()["ok"]  # Cleanup must not release the user's physical button.
    assert device.events == []  # Neither down nor up is allowed for an unowned button.


def test_device_initialization_failure_is_latched_until_reset():
    calls, device = [], FakeMouse()  # Track creation independently from button events.

    def create():
        calls.append("create")  # Count initialization attempts to reject automatic retries.
        if len(calls) == 1:  # Simulate an unavailable native adapter on the first attempt.
            raise OSError("device unavailable")  # Construction failed before any button ownership.
        return device  # Explicit reset permits a later successful initialization.

    controller = EegClickController(device_factory=create)  # Construction remains lazy.
    first = controller.click()  # Latch the failed initialization.
    assert "device unavailable" in first["error"] and not first["cleanup_pending"]  # No button needs release.
    assert controller.click() == first and calls == ["create"]  # Repeated calls cannot silently retry.
    assert controller.reset()["ok"]  # Explicit reset clears the fault after input-free cleanup.
    assert controller.click()["click_id"] == 1  # Failed attempts never consume a successful click ID.
    assert device.events == ["down", "up"]  # Only the recovered, authorized click reaches the device.


@pytest.mark.parametrize("failures,attempted,pending", [
    ({"down": 1}, ["down", "up"], False),  # Ambiguous press failures still require release.
    ({"up": 1}, ["down", "up"], True),  # Rejected release retains button ownership.
    ({"down": 1, "up": 1}, ["down", "up"], True),  # Both failures still allow later cleanup.
])
def test_injection_failure_latches_and_cleanup_never_replays_press(failures, attempted, pending):
    device = FakeMouse()  # All failure cases stay inside the fake adapter.
    device.failures = dict(failures)  # Avoid mutating the parameterized test inputs.
    controller = EegClickController(device_factory=lambda: device)  # Keep native input unreachable.
    result = controller.click()  # Attempt one press and its mandatory paired release.
    assert not result["ok"] and result["cleanup_pending"] is pending  # Preserve accurate cleanup state.
    assert device.events == attempted  # Even a failed down is followed by an up attempt.
    assert controller.click() == result  # A latched failure cannot generate another input event.
    assert device.events == attempted  # In particular, do not replay the press or make another click.
    assert controller.close()["ok"]  # A stop retries only a still-owned release.
    expected = attempted + (["up"] if pending else [])  # Confirmed releases must not be repeated.
    assert device.events == expected  # Cleanup contains no extra down event.
    assert controller.click()["click_id"] == 1  # Explicit reuse starts with the first successful click ID.
    assert device.events == expected + ["down", "up"]  # The next successful request is one normal click.


def test_repeated_release_failure_remains_owned_until_cleanup_succeeds():
    device = FakeMouse()  # Simulate a device that initially refuses every release.
    device.failures["up"] = 3  # Fail the click's up and two explicit cleanup attempts.
    controller = EegClickController(device_factory=lambda: device)  # Never open a real mouse device.
    assert controller.click()["cleanup_pending"]  # The failed initial up must retain ownership.
    assert controller.close()["cleanup_pending"]  # First cleanup failure remains visible.
    assert controller.reset()["cleanup_pending"]  # A second failure must not forget the held button.
    assert controller.click().get("error")  # Pending release blocks any new press.
    assert device.events == ["down", "up", "up", "up"]  # Only cleanup releases were retried.
    assert controller.close() == {"ok": True, "cleanup_pending": False}  # Final up confirms release.
    assert device.events == ["down", "up", "up", "up", "up"]  # Cleanup never repeats the original down.


def test_close_waits_for_inflight_click_and_returns_after_button_release():
    entered, resume, closing, closed = (threading.Event() for _ in range(4))  # Coordinate only fake operations.

    class BlockingMouse(FakeMouse):
        def left_down(self):
            super().left_down()  # Record the in-flight press before allowing stop to race.
            entered.set()  # Tell the test the actuator owns an unresolved press.
            assert resume.wait(2)  # Keep the fake press bounded so failures cannot strand the test.

    device = BlockingMouse()  # The blocked call never reaches native Windows input.
    controller = EegClickController(device_factory=lambda: device)  # Inject the synchronization-aware fake.
    results = {}  # Capture both thread outcomes for assertions in the test thread.

    def stop():
        closing.set()  # Mark the stop request before it waits for the actuator lock.
        results["close"] = controller.close()  # Must wait until the ongoing down/up pair finishes.
        closed.set()  # Successful close may be reported only after release is confirmed.

    clicker = threading.Thread(target=lambda: results.update(click=controller.click()))  # Exercise real locking.
    stopper = threading.Thread(target=stop)  # Make cleanup contend with the unfinished fake click.
    clicker.start()  # Start only the fake button operation.
    try:
        assert entered.wait(1)  # Ensure the press is in flight before requesting close.
        stopper.start()  # Attempt cleanup while click owns the serialization lock.
        assert closing.wait(1)  # Avoid assuming that the cleanup thread was already scheduled.
        assert not closed.wait(.03)  # Close must not claim completion while the press remains unresolved.
    finally:
        resume.set()  # Always unblock the fake operation, including when an assertion fails.
        clicker.join(2)  # Keep worker lifetime within the test even on failure.
        if stopper.ident is not None:  # A failed entry assertion may leave the stopper unstarted.
            stopper.join(2)  # Wait for cleanup once the press has completed.
    assert not clicker.is_alive() and not stopper.is_alive()  # No input work may outlive cleanup.
    assert results["click"]["ok"] and results["close"]["ok"]  # Both operations report confirmed completion.
    assert device.events == ["down", "up"]  # Stop does not double-release or inject a replacement click.
