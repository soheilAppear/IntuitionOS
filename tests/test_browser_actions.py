"""Browser requests preserve destinations and launch without a command shell."""

import sys
from types import SimpleNamespace

import pytest

from core import os_sandbox
from core.os_intents import _try_os_intent, is_app_command


@pytest.mark.parametrize(
    "phrase,browser",
    [
        ("open chrome and go to google.com", "chrome"),
        ("go to chrome open it and go to google.com", "chrome"),
        ("go to chrome and open google.com", "chrome"),
        ("open chrome and go to google.com!", "chrome"),
        ("openchrome for me and go to google.com", "chrome"),
        ("go to cvhrome open it and go to google.com", "chrome"),
        ("open google.com in chrome", "chrome"),
        ("please open Google Chrome and visit google.com", "chrome"),
        ("Can you please open google.com using Chrome for me", "chrome"),
        ("launch edge then navigate to google.com", "edge"),
        ("open google.com with Microsoft Edge", "edge"),
        ("start Firefox and open google.com", "firefox"),
        ("open google.com in Mozilla Firefox", "firefox"),
        ("open google.com", "default"),
        ("go to google.com", "default"),
        ("goto google.com", "default"),
        ("goto to google.com", "default"),
        ("visit google.com", "default"),
        ("open browser and go to google.com", "default"),
    ],
)
def test_complete_browser_requests_are_recognized_and_protected(phrase, browser):
    assert _try_os_intent(phrase) == (
        "os_open_url", {"url": "https://google.com", "browser": browser}
    )
    assert is_app_command(phrase)


def test_browser_intent_preserves_case_sensitive_destination():
    destination = "https://Example.com/Case/Sensitive?q=Mixed&key=AbC#PartOne"
    assert _try_os_intent(f"open Chrome and go to {destination}") == (
        "os_open_url", {"url": destination, "browser": "chrome"}
    )


@pytest.mark.parametrize("destination", [
    "https://example.com/Path!", "https://example.com/Path?", "https://example.com?q=Hi!",
])
def test_url_path_and_query_punctuation_is_preserved(destination):
    assert _try_os_intent(f"open Chrome and go to {destination}")[1]["url"] == destination


@pytest.mark.parametrize("phrase", [
    "open chrome and go to google.com and take a screenshot",
    "open google.com and then check the battery",
    "go to chrome and open google.com and read clipboard",
])
def test_compound_browser_request_cannot_execute_only_its_later_os_action(phrase):
    assert _try_os_intent(phrase) is None
    assert is_app_command(phrase)  # Preserve the whole request for the AI.


@pytest.mark.parametrize("phrase", [
    "git commit -m 'open chrome and go to google.com'",
    "python script.py --url https://google.com",
    "curl https://google.com",
    "chrome https://google.com --incognito",
    "openai chrome and go to google.com",
    "echo 'go to google.com'",
    "go run main.go",
])
def test_browser_grammar_does_not_hijack_shell_arguments(phrase):
    assert not is_app_command(phrase)


@pytest.mark.parametrize("phrase,name", [
    ("open chrome", "chrome"), ("open notepad", "notepad"),
    ("launch firefox", "firefox"), ("open calc.exe", "calc.exe"),
])
def test_existing_app_requests_are_unchanged(phrase, name):
    assert _try_os_intent(phrase) == ("os_open_app", {"name": name})


@pytest.mark.parametrize("url,expected", [
    ("google.com", "https://google.com"),
    ("example.com/Case?q=One&b=Two#Here", "https://example.com/Case?q=One&b=Two#Here"),
    ("HTTP://Example.com/Path", "HTTP://Example.com/Path"),
    ("http://localhost:8080/Status", "http://localhost:8080/Status"),
    ("localhost:8080/Status", "https://localhost:8080/Status"),
    ("[::1]:8080/Status", "https://[::1]:8080/Status"),
])
def test_url_normalization(url, expected):
    assert os_sandbox.normalize_url(url) == expected


@pytest.mark.parametrize("url", [
    "", None, "javascript:alert(1)", "file:///C:/Windows/notepad.exe",
    "ftp://example.com", "https://", "https:///path", "https://example.com:99999",
    "https://example.com:0", "https://user:secret@example.com",
    "https://example.com\ncalc.exe", "https://example.com/a b",
    "https://example.com\\evil", "https://example.com\"", "https://-bad.com",
    "https://example..com", "https://example.com&calc.exe",
])
def test_invalid_url_is_rejected_before_launch(url, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid URL must never launch a browser")

    monkeypatch.setattr(os_sandbox.subprocess, "Popen", unexpected)
    monkeypatch.setattr(os_sandbox.os, "startfile", unexpected, raising=False)
    monkeypatch.setattr(os_sandbox.webbrowser, "open", unexpected)
    with pytest.raises(ValueError):
        os_sandbox.normalize_url(url)
    assert "error" in os_sandbox.open_url(url, "chrome")


def test_explicit_browser_receives_url_as_one_literal_argument(monkeypatch):
    destination = "https://example.com/Case?q=A&next=$(calc)&other=%USERPROFILE%"
    calls = []
    monkeypatch.setattr(os_sandbox, "_find_exe", lambda exe: "C:/Apps/Chrome/chrome.exe")
    monkeypatch.setattr(os_sandbox.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    result = os_sandbox.open_url(destination, "chrome")
    assert calls == [((["C:/Apps/Chrome/chrome.exe", destination],), {"shell": False})]
    assert result["status"] == "open_requested"
    assert result["url"] == destination
    assert "not verified" in result["message"]


def test_missing_explicit_browser_reports_error_without_switching_browser(monkeypatch):
    monkeypatch.setattr(os_sandbox, "_find_exe", lambda exe: None)
    monkeypatch.setattr(os_sandbox.webbrowser, "open", lambda *a, **k: pytest.fail("No fallback"))
    monkeypatch.setattr(os_sandbox.os, "startfile", lambda *a: pytest.fail("No fallback"), raising=False)
    result = os_sandbox.open_url("google.com", "chrome")
    assert "Could not find chrome" in result["error"]


def test_browser_launch_failure_is_visible(monkeypatch):
    monkeypatch.setattr(os_sandbox, "_find_exe", lambda exe: "C:/Apps/Chrome/chrome.exe")

    def fail(*args, **kwargs):
        raise OSError("access denied")

    monkeypatch.setattr(os_sandbox.subprocess, "Popen", fail)
    assert "access denied" in os_sandbox.open_url("google.com", "chrome")["error"]


def test_unknown_browser_is_rejected(monkeypatch):
    monkeypatch.setattr(os_sandbox, "_find_exe", lambda exe: pytest.fail("Invalid browser"))
    assert "error" in os_sandbox.open_url("google.com", "cmd.exe")


def test_default_browser_uses_windows_url_handler(monkeypatch):
    calls = []
    monkeypatch.setattr(os_sandbox.sys, "platform", "win32")
    monkeypatch.setattr(os_sandbox.os, "startfile", calls.append, raising=False)
    result = os_sandbox.open_url("google.com")
    assert calls == ["https://google.com"]
    assert result["browser"] == "default"
    assert result["status"] == "open_requested"


def test_default_browser_failure_is_visible_on_other_platforms(monkeypatch):
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox.webbrowser, "open", lambda *a, **k: False)
    assert "error" in os_sandbox.open_url("google.com")


@pytest.mark.parametrize("registered_hive,registered_view", [(1, 64), (2, 32)])
def test_registered_windows_app_is_resolved_without_a_shell(monkeypatch, registered_hive, registered_view):
    executable = r"C:\Program Files\Microsoft Office\Root\Office16\WINWORD.EXE"
    opened = []

    class RegistryKey:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def open_key(hive, key, reserved, access):
        opened.append((hive, key, access))
        if hive == registered_hive and access == 4 | registered_view:
            return RegistryKey()
        raise FileNotFoundError()

    registry = SimpleNamespace(
        HKEY_CURRENT_USER=1, HKEY_LOCAL_MACHINE=2,
        KEY_READ=4, KEY_WOW64_64KEY=64, KEY_WOW64_32KEY=32,
        OpenKey=open_key, QueryValueEx=lambda key, name: (executable, 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr(os_sandbox.sys, "platform", "win32")
    monkeypatch.setattr(os_sandbox.shutil, "which", lambda name: None)
    monkeypatch.setattr(os_sandbox.os.path, "isfile", lambda path: path == executable)
    assert os_sandbox._find_exe("WINWORD.EXE") == executable
    assert all(key.endswith(r"App Paths\WINWORD.EXE") for _, key, _ in opened)


def test_unknown_app_cannot_report_success_from_launching_cmd(monkeypatch):
    monkeypatch.setattr(os_sandbox.sys, "platform", "win32")
    monkeypatch.setattr(os_sandbox, "_find_exe", lambda exe: None)
    monkeypatch.setattr(os_sandbox.subprocess, "Popen", lambda *a, **k: pytest.fail("No cmd fallback"))
    result = os_sandbox.open_app("missing-app & calc.exe")
    assert "Could not find" in result["error"]


def test_url_cannot_be_misreported_as_application_launch(monkeypatch):
    monkeypatch.setattr(os_sandbox, "_find_exe", lambda exe: pytest.fail("URL is not an app"))
    assert "open URL action" in os_sandbox.open_app("https://google.com")["error"]
