"""Generated content belongs in ignored output, including direct and undo paths."""

from pathlib import Path

import pytest

from core import actions as actions_mod
from core.capabilities import capabilities, gate, jail_path, set_safe_mode


@pytest.mark.parametrize("given,relative", [
    ("hello.py", "hello.py"),
    ("scripts/hello.py", "scripts/hello.py"),
    ("CreatedFolder/hello.py", "hello.py"),
    ("CreatedFolder/scripts/hello.py", "scripts/hello.py"),
    ("./CreatedFolder/hello.py", "hello.py"),
])
def test_generated_paths_rebase_once_without_gate_side_effects(project, given, relative):
    decision = gate(capabilities.get("write_file"), {"path": given, "text": "print('hello')\n"},
                    confidence=1.0, actor="user")
    assert decision.allowed, decision.reason
    assert Path(decision.args["path"]) == project / "CreatedFolder" / relative
    assert not (project / "CreatedFolder").exists()


@pytest.mark.parametrize("safe", [True, False])
def test_model_can_write_python_and_read_back_exact_contents(project, wired, safe):
    actions, journal, _ = wired
    set_safe_mode(safe)
    code = "def greeting(name):\n    return f'Hello, {name}!'\n\nprint(greeting('Soheil'))\n"
    result = actions.dispatch("write_file", {"path": "scripts/hello.py", "text": code},
                              actor="model", confidence=1.0)
    target = project / "CreatedFolder" / "scripts" / "hello.py"
    assert result["ok"] and Path(result["path"]) == target
    assert target.read_text(encoding="utf-8") == code
    assert actions.call("read_file", path=result["path"])["text"] == code
    assert journal.recent()[0]["args"]["path"] == str(target)
    assert not (project / "scripts").exists()


def test_absolute_output_path_is_accepted_and_undo_uses_same_path(project, wired):
    actions, journal, _ = wired
    target = project / "CreatedFolder" / "note.txt"
    result = actions.call("write_file", path=str(target), text="local output")
    assert result["ok"]
    assert journal.last_undoable()["undo"]["path"] == str(target)
    assert actions_mod.undo_last()["ok"]
    assert not target.exists()


@pytest.mark.parametrize("path", [
    "../outside.txt", "CreatedFolder/../outside.txt", "nested/../../outside.txt",
    ".", "CreatedFolder", "a.txt:stream", "CON.txt", "trailing.txt.",
])
def test_invalid_generated_paths_are_denied_before_output_creation(project, wired, path):
    actions, journal, _ = wired
    result = actions.call("write_file", path=path, text="must not be written")
    assert result.get("denied"), result
    assert not (project / "CreatedFolder").exists()
    assert journal.last_undoable() is None


def test_absolute_project_source_path_is_denied_even_with_safe_mode_off(project, wired):
    actions, _, _ = wired
    source = project / "source.py"
    source.write_text("keep source", encoding="utf-8")
    set_safe_mode(False)
    result = actions.call("write_file", path=str(source), text="replace source")
    assert result.get("denied")
    assert source.read_text(encoding="utf-8") == "keep source"
    assert not (project / "CreatedFolder").exists()


def test_direct_write_implementation_enforces_output_scope(project):
    outside = project / "outside.txt"
    assert "error" in actions_mod.write_file(str(outside), "no")
    assert not outside.exists()
    result = actions_mod.write_file("direct.txt", "yes")
    assert result["ok"]
    assert Path(result["path"]) == project / "CreatedFolder" / "direct.txt"


def _symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is unavailable for this user")


@pytest.mark.parametrize("kind", ["root", "directory", "leaf"])
def test_redirected_output_cannot_overwrite_project_source(project, wired, kind):
    actions, _, _ = wired
    outside = project / "source"
    outside.mkdir()
    source = outside / "keep.py"
    source.write_text("keep", encoding="utf-8")
    output = project / "CreatedFolder"
    if kind == "root":
        _symlink_or_skip(output, outside, directory=True)
        requested = "keep.py"
    else:
        output.mkdir()
        if kind == "directory":
            _symlink_or_skip(output / "redirect", outside, directory=True)
            requested = "redirect/keep.py"
        else:
            _symlink_or_skip(output / "keep.py", source)
            requested = "keep.py"
    assert actions.call("write_file", path=requested, text="overwrite").get("denied")
    assert source.read_text(encoding="utf-8") == "keep"


def test_undo_refuses_a_parent_redirected_since_the_write(project, wired):
    actions, journal, _ = wired
    result = actions.call("write_file", path="nested/note.txt", text="generated")
    target = Path(result["path"])
    original = target.parent.with_name("original")
    target.parent.rename(original)
    outside = project / "source"
    outside.mkdir()
    source = outside / "note.txt"
    source.write_text("keep", encoding="utf-8")
    _symlink_or_skip(target.parent, outside, directory=True)
    assert "error" in actions_mod.undo_last()
    assert source.read_text(encoding="utf-8") == "keep"
    assert journal.last_undoable() is not None


def test_read_scope_remains_project_scoped(project):
    path, problem = jail_path("README.md", "project")
    assert problem is None and Path(path) == project / "README.md"


def test_speculative_write_does_not_create_output_directory(project, wired):
    actions, _, _ = wired
    result = actions.dispatch("write_file", {"path": "guess.py", "text": "print('no')"},
                              actor="anticipator", confidence=1.0)
    assert result.get("denied")
    assert not (project / "CreatedFolder").exists()
