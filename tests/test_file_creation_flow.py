"""Natural-language file creation through the real resolver, gate, and journal."""

from pathlib import Path

import pytest

from core import user_files
from core.actions import actions
from core.capabilities import capabilities, is_safe_mode, set_safe_mode
from core.command_resolver import CommandResolver, IntuitionCommandProvider, StaticCommandProvider
from interface import server, terminal
from test_hud_resolution import hud as hud
from test_hud_resolution import preview, receive, submit
from test_terminal_resolution import run_main


@pytest.fixture
def user_folders(hud, monkeypatch):
    folders = {"project": Path.cwd()}
    for name in ("desktop", "documents", "downloads"):
        folders[name] = Path.cwd() / "Redirected User Folders" / name.capitalize()
        folders[name].mkdir(parents=True)
    monkeypatch.setattr(user_files, "_folder_path", lambda name: folders[name])
    server._state["resolver"].providers.append(StaticCommandProvider(
        ["make", "makecab", "mkdir"], case_sensitive=False,
    ))
    monkeypatch.setattr(server._state["brain"], "step", lambda *a, **kw: pytest.fail("Empty file request reached the model"))
    return folders


@pytest.mark.parametrize("safe", [True, False])
def test_reported_request_creates_desktop_file_without_makecab_and_can_undo(hud, user_folders, safe):
    client, _ = hud
    set_safe_mode(safe)
    target = user_folders["desktop"] / "1.py"
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        shown = preview(ws, "make a new file 1.py in desktop directory")
        assert shown["namespace"] == "intuitionos" and not shown["candidates"]
        submit(ws, shown, index=None)
        assert receive(ws, "thinking")["text"] == "Creating file…"
        assert str(target) in receive(ws, "reply")["text"]
        assert target.is_file() and target.read_bytes() == b""
        assert is_safe_mode() is safe
        assert server.get_journal().recent()[0]["capability"] == "create_empty_file"
        ws.send_json({"type": "input", "text": "/undo"})
        receive(ws, "reply")
        assert not target.exists()


def test_existing_desktop_file_is_preserved_with_clear_reply(hud, user_folders):
    client, _ = hud
    target = user_folders["desktop"] / "1.py"
    target.write_text("print('keep this')", encoding="utf-8")
    with client.websocket_connect("/ws") as ws:
        receive(ws, "status")
        submit(ws, preview(ws, "make a new file 1.py in desktop directory"), index=None)
        assert "already exists" in receive(ws, "reply")["text"]
    assert target.read_text(encoding="utf-8") == "print('keep this')"
    assert server.get_journal().recent()[0]["outcome"] == "error"


def test_edited_created_file_cannot_be_deleted_by_journal_undo(hud, user_folders):
    target = user_folders["desktop"] / "1.py"
    assert actions.call("create_empty_file", name="1.py", directory="desktop")["ok"]
    target.write_text("print('new work')", encoding="utf-8")
    result = server.get_journal().undo_last(capabilities)
    assert "edited or replaced" in result["error"]
    assert target.read_text(encoding="utf-8") == "print('new work')"
    assert server.get_journal().recent()[0]["undone_at"] is None


@pytest.mark.parametrize("args", [
    {"name": "../outside.py", "directory": "desktop"},
    {"name": "1.py", "directory": "C:/Windows"},
    {"name": "1.py", "directory": "desktop", "overwrite": True},
])
def test_file_capability_rejects_paths_or_undeclared_options_before_creation(hud, user_folders, args):
    assert actions.dispatch("create_empty_file", args)["denied"]
    assert not list(user_folders["desktop"].iterdir())


def test_speculative_file_creation_is_denied(hud, user_folders):
    assert actions.dispatch("create_empty_file", {"name": "1.py", "directory": "desktop"}, actor="anticipator")["denied"]
    assert not (user_folders["desktop"] / "1.py").exists()


def test_terminal_routes_same_file_request_directly(monkeypatch, project, wired):
    _, _, memory = wired
    resolver = CommandResolver([
        IntuitionCommandProvider(), StaticCommandProvider(["makecab", "make"]),
    ], shell="cmd")
    calls = []
    monkeypatch.setattr(terminal, "run_action", lambda action_name, **args: (
        calls.append((action_name, args)) or {"ok": True, "path": "test Desktop/1.py"}
    ))
    run_main(monkeypatch, memory, ["make a new file 1.py in desktop directory", "/exit"], resolver)
    assert calls == [("create_empty_file", {"name": "1.py", "directory": "desktop"})]
