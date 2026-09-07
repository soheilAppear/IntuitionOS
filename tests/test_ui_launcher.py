"""Launcher lifecycle regressions; no tests start or stop real processes."""

import subprocess
from unittest.mock import Mock

import pytest

import start_ui as launcher


class Child:
    def __init__(self, codes=(None,)):
        self.pid = 4242
        self.codes = iter(codes)
        self.code = None
        self.terminated = self.killed = self.waited = False

    def poll(self):
        self.code = next(self.codes, self.code)
        return self.code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


@pytest.fixture(autouse=True)
def no_live_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Launcher test attempted a real process/network operation")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher.socket, "create_connection", forbidden)
    monkeypatch.setattr(launcher.http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    # Ordinary lifecycle cases exercise portable termination. Windows tree
    # cleanup has separate tests with a fully mocked taskkill process.
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")


def test_windows_prefers_native_electron_over_cmd_shim(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(launcher, "UI_DIR", str(tmp_path))
    shim = tmp_path / "node_modules" / ".bin" / "electron.cmd"
    native = tmp_path / "node_modules" / "electron" / "dist" / "electron.exe"
    shim.parent.mkdir(parents=True)
    native.parent.mkdir(parents=True)
    shim.write_text("shim")
    native.write_text("native")
    assert launcher._find_electron() == str(native)
    native.unlink()
    assert launcher._find_electron() is None


def test_windows_npm_install_uses_explicit_cmd_and_rechecks_binary(monkeypatch):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    binaries = iter([None, "electron.exe"])
    monkeypatch.setattr(launcher, "_find_electron", lambda: next(binaries))
    monkeypatch.setattr(
        launcher.shutil, "which", lambda name: r"C:\Program Files\nodejs\npm.cmd"
    )
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher._ensure_electron() == "electron.exe"
    assert run.call_args.args == (
        '"C:\\Windows\\System32\\cmd.exe" /d /s /c ""C:\\Program Files\\nodejs\\npm.cmd" install"',
    )
    assert run.call_args.kwargs == {"cwd": launcher.UI_DIR}


@pytest.mark.parametrize("healthy", [True, False])
def test_existing_listener_is_diagnosed_without_spawning_or_stopping(
    healthy, monkeypatch, capsys
):
    monkeypatch.setattr(launcher, "_port_in_use", lambda: True)
    monkeypatch.setattr(launcher, "_backend_healthy", lambda: healthy)
    assert launcher.main() == 1
    output = capsys.readouterr().out
    assert "7432" in output
    assert ("already responding" if healthy else "occupied") in output


def test_readiness_waits_for_health_instead_of_a_fixed_delay(monkeypatch):
    server = Child()
    responses = iter([False, False, True])
    monkeypatch.setattr(launcher, "_backend_healthy", lambda: next(responses))
    launcher._wait_for_backend(server)
    assert not server.terminated


def test_readiness_rejects_exited_child_even_if_another_health_endpoint_responds(
    monkeypatch,
):
    server = Child([1])
    health = Mock(return_value=True)
    monkeypatch.setattr(launcher, "_backend_healthy", health)
    with pytest.raises(RuntimeError, match="exited during startup"):
        launcher._wait_for_backend(server)
    health.assert_not_called()


def test_readiness_timeout_is_actionable(monkeypatch):
    times = iter([0, 0, 2])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(launcher, "_backend_healthy", lambda: False)
    with pytest.raises(RuntimeError, match="did not become ready"):
        launcher._wait_for_backend(Child(), timeout=1)


@pytest.mark.parametrize("shell", ["powershell", "pwsh"])
def test_powershell_cold_discovery_can_finish_before_launcher_deadline(shell, monkeypatch):
    monkeypatch.setenv("INTUITION_SHELL", shell)
    times = iter([0, 0, 61])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    responses = iter([False, True])
    monkeypatch.setattr(launcher, "_backend_healthy", lambda: next(responses))
    launcher._wait_for_backend(Child())


def test_cmd_keeps_normal_startup_deadline(monkeypatch):
    monkeypatch.setenv("INTUITION_SHELL", "cmd")
    times = iter([0, 0, 61])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(launcher, "_backend_healthy", lambda: False)
    with pytest.raises(RuntimeError, match="within 60 seconds"):
        launcher._wait_for_backend(Child())


def prepare_main(monkeypatch, children):
    monkeypatch.setattr(launcher, "_check_port_available", lambda: None)
    monkeypatch.setattr(launcher, "_ensure_electron", lambda: "electron.exe")
    monkeypatch.setattr(launcher, "_wait_for_backend", lambda server: None)
    iterator = iter(children)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: next(iterator))


def test_electron_spawn_failure_always_cleans_up_backend(monkeypatch):
    server = Child()
    prepare_main(monkeypatch, [server])
    calls = iter([server, OSError("bad executable")])

    def spawn(*args, **kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    assert launcher.main() == 1
    assert server.terminated and server.waited


def test_failed_startup_never_opens_electron_and_cleans_up(monkeypatch):
    server = Child()
    prepare_main(monkeypatch, [server])

    def failed(server):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(launcher, "_wait_for_backend", failed)
    assert launcher.main() == 1
    assert server.terminated


def test_backend_death_closes_owned_electron(monkeypatch):
    server, electron = Child([1]), Child()
    prepare_main(monkeypatch, [server, electron])
    assert launcher.main() == 1
    assert electron.terminated and electron.waited
    assert not server.terminated


def test_hud_quit_stops_backend(monkeypatch):
    server, electron = Child(), Child([0])
    prepare_main(monkeypatch, [server, electron])
    assert launcher.main() == 0
    assert server.terminated and server.waited


def test_electron_node_mode_is_removed_only_from_the_hud_child(monkeypatch):
    server, electron = Child(), Child([0])
    prepare_main(monkeypatch, [server, electron])
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    monkeypatch.setenv("INTUITION_LAUNCHER_TEST", "preserved")
    children = iter([server, electron])
    launches = []

    def spawn(*args, **kwargs):
        launches.append(kwargs)
        return next(children)

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    assert launcher.main() == 0
    assert "env" not in launches[0]  # Python backend inherits the original env.
    assert "ELECTRON_RUN_AS_NODE" not in launches[1]["env"]
    assert launches[1]["env"]["INTUITION_LAUNCHER_TEST"] == "preserved"
    assert launcher.os.environ["ELECTRON_RUN_AS_NODE"] == "1"


def test_keyboard_interrupt_stops_both_owned_children(monkeypatch):
    server, electron = Child(), Child()
    prepare_main(monkeypatch, [server, electron])

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher, "_monitor", interrupted)
    assert launcher.main() == 0
    assert server.terminated and electron.terminated


def test_unresponsive_child_is_killed_and_reaped():
    child = Child()
    child.wait = Mock(side_effect=[subprocess.TimeoutExpired("child", 5), 0])
    launcher._stop_process(child)
    assert child.terminated and child.killed
    assert child.wait.call_count == 2


def test_windows_cleanup_stops_only_owned_pid_tree_before_reaping(monkeypatch):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    child = Child()
    launcher._stop_process(child)
    assert run.call_args.args == (["taskkill.exe", "/PID", "4242", "/T", "/F"],)
    assert run.call_args.kwargs["timeout"] == 5
    assert "shell" not in run.call_args.kwargs
    assert not child.terminated and not child.killed
    assert child.waited


def test_windows_tree_cleanup_failure_is_reported_without_losing_parent_first(
    monkeypatch, capsys
):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        Mock(return_value=Mock(returncode=1, stderr="Access denied")),
    )
    child = Child()
    launcher._stop_process(child)
    assert "Access denied" in capsys.readouterr().out
    assert not child.terminated and not child.killed


@pytest.mark.parametrize(
    "status,payload,expected",
    [
        (200, b'{"ok": true, "version": "1.0"}', True),
        (200, b'{"ok": false, "version": "1.0"}', False),
        (200, b'{"ok": true}', False),
        (200, b"not json", False),
        (503, b"{}", False),
    ],
)
def test_health_probe_requires_expected_response_and_closes_connection(
    status, payload, expected, monkeypatch
):
    connection = Mock()
    connection.getresponse.return_value = Mock(
        status=status, read=Mock(return_value=payload)
    )
    monkeypatch.setattr(
        launcher.http.client, "HTTPConnection", lambda *a, **k: connection
    )
    assert launcher._backend_healthy() is expected
    connection.request.assert_called_once_with("GET", "/health")
    connection.close.assert_called_once()
