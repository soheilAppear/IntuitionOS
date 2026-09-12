"""File creation uses real named folders and never overwrites user work."""

import os
from pathlib import Path

import pytest

from core import user_files


@pytest.fixture(autouse=True)
def isolated_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_create_project_file_and_undo_without_shell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = user_files.create_empty_file("1.py")
    assert result["ok"] and result["created"]
    assert result["bytes"] == 0
    target = tmp_path.resolve() / "CreatedFolder" / "1.py"
    assert Path(result["path"]) == target
    assert target.read_bytes() == b""
    undone = user_files.undo_created_file(result["undo"])
    assert undone["deleted"]
    assert not target.exists()


def test_project_output_matches_current_project_scope(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = user_files.create_empty_file("notes.txt", "project")
    assert result["ok"]
    assert Path(result["path"]) == elsewhere.resolve() / "CreatedFolder" / "notes.txt"
    assert not (tmp_path / "CreatedFolder").exists()


def test_validation_does_not_create_the_output_directory(tmp_path):
    assert user_files.resolve_file_target("notes.txt") == tmp_path.resolve() / "CreatedFolder" / "notes.txt"
    assert list(tmp_path.iterdir()) == []


def test_created_folder_file_is_not_replaced(tmp_path):
    output = tmp_path / "CreatedFolder"
    output.write_text("keep this file", encoding="utf-8")
    assert "error" in user_files.create_empty_file("notes.txt")
    assert output.read_text(encoding="utf-8") == "keep this file"


def test_created_folder_symlink_cannot_redirect_output(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(tmp_path / "CreatedFolder", outside, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        user_files.resolve_file_target("notes.txt")
    assert "error" in user_files.create_empty_file("notes.txt")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("directory", ["desktop", "documents", "downloads"])
def test_windows_named_folder_uses_redirected_known_folder(tmp_path, monkeypatch, directory):
    redirected = tmp_path / "OneDrive - Example" / directory.capitalize()
    redirected.mkdir(parents=True)
    requests = []

    def known_folder(which):
        requests.append(which)
        return redirected

    monkeypatch.setattr(user_files.sys, "platform", "win32")
    monkeypatch.setattr(user_files, "_windows_known_folder", known_folder)
    result = user_files.create_empty_file("notes.py", directory)
    assert requests == [directory]
    assert result["ok"]
    assert Path(result["path"]) == redirected.resolve() / "notes.py"


def test_unknown_windows_folder_never_falls_back_to_home(tmp_path, monkeypatch):
    monkeypatch.setattr(user_files.sys, "platform", "win32")
    monkeypatch.setattr(user_files.Path, "home", lambda: tmp_path)

    def unavailable(_):
        raise OSError("Windows could not locate your desktop folder")

    monkeypatch.setattr(user_files, "_windows_known_folder", unavailable)
    result = user_files.create_empty_file("1.py", "desktop")
    assert "could not locate" in result["error"]
    assert not (tmp_path / "Desktop").exists()


def test_posix_named_folder(tmp_path, monkeypatch):
    (tmp_path / "Desktop").mkdir()
    monkeypatch.setattr(user_files.sys, "platform", "linux")
    monkeypatch.setattr(user_files.Path, "home", lambda: tmp_path)
    assert user_files.resolve_file_target("1.py", "desktop") == tmp_path.resolve() / "Desktop" / "1.py"


@pytest.mark.parametrize("name", [
    "", " ", " 1.py", "1.py ", ".", "..", "../1.py", "..\\1.py",
    "/1.py", "C:\\1.py", "folder/1.py", "1.py:stream", "1.py\x00",
    "1\n.py", "1.py.", "1?.py", "1*.py", "1|2.py", '1".py',
    "CON", "con.py", "NUL.txt", "AUX", "PRN", "COM1.py", "LPT9.txt",
    "COM¹.py", "LPT².py", "CON .py", "CONIN$", "CONOUT$.txt", "a" * 256,
])
def test_invalid_filename_cannot_escape_or_reach_device(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        user_files.resolve_file_target(name)
    assert "error" in user_files.create_empty_file(name)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("directory", ["../desktop", "C:\\", "/tmp", "Desktop", "", None])
def test_only_named_directories_are_allowed(directory, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        user_files.resolve_file_target("1.py", directory)
    assert not (tmp_path / "1.py").exists()


def test_missing_directory_is_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "UnavailableDesktop"
    monkeypatch.setattr(user_files, "_folder_path", lambda _: missing)
    result = user_files.create_empty_file("1.py", "desktop")
    assert "unavailable" in result["error"]
    assert not missing.exists()


def test_existing_file_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = user_files.created_folder_path(create=True) / "1.py"
    target.write_text("print('keep me')", encoding="utf-8")
    result = user_files.create_empty_file("1.py")
    assert "already exists" in result["error"]
    assert "undo" not in result
    assert target.read_text(encoding="utf-8") == "print('keep me')"


def test_existing_directory_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = user_files.created_folder_path(create=True) / "1.py"
    target.mkdir()
    assert "error" in user_files.create_empty_file("1.py")
    assert target.is_dir()


def test_file_created_after_validation_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_resolve = user_files.resolve_file_target

    def resolve_then_another_app_creates(name, directory):
        user_files.created_folder_path(create=True)
        target = original_resolve(name, directory)
        target.write_text("another application's work", encoding="utf-8")
        return target

    monkeypatch.setattr(user_files, "resolve_file_target", resolve_then_another_app_creates)
    result = user_files.create_empty_file("1.py")
    assert "already exists" in result["error"]
    assert (tmp_path / "CreatedFolder" / "1.py").read_text(encoding="utf-8") == "another application's work"


def _symlink_or_skip(link, target, *, target_is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is unavailable for this user")


@pytest.mark.parametrize("outside_exists", [False, True])
def test_leaf_symlink_cannot_write_outside_folder(tmp_path, monkeypatch, outside_exists):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    output = user_files.created_folder_path(create=True)
    outside = tmp_path / "outside.py"
    if outside_exists:
        outside.write_text("outside", encoding="utf-8")
    _symlink_or_skip(output / "1.py", outside)
    monkeypatch.chdir(project)
    assert "error" in user_files.create_empty_file("1.py")
    assert outside.exists() == outside_exists
    if outside_exists:
        assert outside.read_text(encoding="utf-8") == "outside"


def test_existing_symlink_inside_folder_is_never_followed(tmp_path, monkeypatch):
    output = user_files.created_folder_path(create=True)
    target = output / "other.py"
    target.write_text("keep", encoding="utf-8")
    _symlink_or_skip(output / "1.py", target)
    monkeypatch.chdir(tmp_path)
    assert "error" in user_files.create_empty_file("1.py")
    assert target.read_text(encoding="utf-8") == "keep"


def test_undo_refuses_edited_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = user_files.create_empty_file("1.py")
    target = Path(result["path"])
    target.write_text("valuable new code", encoding="utf-8")
    with pytest.raises(ValueError, match="edited or replaced"):
        user_files.undo_created_file(result["undo"])
    assert target.read_text(encoding="utf-8") == "valuable new code"


def test_undo_refuses_file_edited_back_to_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = user_files.create_empty_file("1.py")
    target = Path(result["path"])
    info = target.stat()
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    assert target.stat().st_size == 0
    with pytest.raises(ValueError, match="edited or replaced"):
        user_files.undo_created_file(result["undo"])
    assert target.exists()


def test_undo_refuses_replacement_empty_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = user_files.create_empty_file("1.py")
    target = Path(result["path"])
    target.rename(tmp_path / "original.py")
    target.touch()
    with pytest.raises(ValueError, match="edited or replaced"):
        user_files.undo_created_file(result["undo"])
    assert target.exists()
    assert (tmp_path / "original.py").exists()


def test_undo_refuses_replacement_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = user_files.create_empty_file("1.py")
    target = Path(result["path"])
    target.unlink()
    other = target.parent / "other.py"
    other.touch()
    _symlink_or_skip(target, other)
    with pytest.raises(ValueError, match="replaced"):
        user_files.undo_created_file(result["undo"])
    assert target.is_symlink()
    assert other.exists()


def test_undo_refuses_missing_identity(tmp_path):
    with pytest.raises(ValueError, match="missing file identity"):
        user_files.undo_created_file({"path": str(tmp_path / "1.py")})


def test_undo_refuses_changed_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = user_files.create_empty_file("1.py")
    result["undo"]["parent"] = str(tmp_path.parent)
    with pytest.raises(ValueError, match="invalid saved path"):
        user_files.undo_created_file(result["undo"])
    assert Path(result["path"]).exists()
