"""Create a single empty file in a named user folder without a shell.

These implementations are called through the capability registry. They never
overwrite an existing entry. Default output belongs in the project's ignored
CreatedFolder directory; only that directory is created automatically. Windows'
Known Folder API supplies the real location of explicitly named user folders.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import stat
import sys
import uuid


_FOLDER_IDS = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
}
_RESERVED = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]|CONIN\$|CONOUT\$)$", re.I)


def _windows_known_folder(directory: str) -> Path:
    """Ask Windows where this user's folder lives; never guess after a failure."""
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

    folder_id = GUID.from_buffer_copy(uuid.UUID(_FOLDER_IDS[directory]).bytes_le)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    get_folder = shell32.SHGetKnownFolderPath
    get_folder.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE,
                           ctypes.POINTER(ctypes.c_void_p)]
    get_folder.restype = ctypes.c_long
    free = ole32.CoTaskMemFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    pointer = ctypes.c_void_p()
    try:
        # Zero flags query the current path without creating the directory.
        status = get_folder(ctypes.byref(folder_id), 0, None, ctypes.byref(pointer))
        if status != 0 or not pointer.value:
            raise OSError(f"Windows could not locate your {directory} folder "
                          f"(HRESULT 0x{status & 0xFFFFFFFF:08X}).")
        return Path(ctypes.wstring_at(pointer.value))
    finally:
        if pointer.value:
            free(pointer)


def _folder_path(directory: str) -> Path:
    if directory == "project":
        return created_folder_path()
    if directory not in _FOLDER_IDS:
        raise ValueError("Choose project, desktop, documents, or downloads as the directory.")
    if sys.platform == "win32":
        return _windows_known_folder(directory)
    return Path.home() / directory.capitalize()


def created_folder_path(*, create: bool = False) -> Path:
    """Locate output in the current project, refusing redirected folders.

    Validation callers leave ``create`` false so permission checks have no
    filesystem side effects. Execution may create this one direct child of the
    project, never an arbitrary hierarchy supplied in a model argument. The
    launcher sets cwd to the repository, matching the capability project scope.
    """
    project = Path.cwd().resolve(strict=True)
    if not project.is_dir():
        raise ValueError(f"The project location is not a directory: {project}")
    folder = project / "CreatedFolder"
    if create:
        folder.mkdir(exist_ok=True)
    try:
        info = folder.lstat()
    except FileNotFoundError:
        return folder
    if _is_redirect(info) or not stat.S_ISDIR(info.st_mode) or folder.resolve() != folder:
        raise ValueError("CreatedFolder must be a regular directory inside the IntuitionOS project.")
    return folder


def _validate_filename(name: str) -> None:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Give a filename such as 1.py, without leading or trailing spaces.")
    if name in {".", ".."} or any(c in name for c in '/\\<>:"|?*'):
        raise ValueError("Give one filename, without a path, path separators, or reserved characters.")
    if any(ord(c) < 32 or ord(c) == 127 for c in name) or name.endswith("."):
        raise ValueError("The filename cannot contain control characters or end with a dot.")
    if len(name.encode("utf-16-le")) // 2 > 255:
        raise ValueError("The filename is too long; use at most 255 characters.")
    # Windows device names remain reserved even with extensions or spaces before
    # the extension, and even when a command is tested on another operating system.
    if _RESERVED.fullmatch(name.split(".", 1)[0].rstrip(" ")):
        raise ValueError(f"{name!r} is a reserved Windows device name; choose another filename.")


def resolve_file_target(name: str, directory: str = "project") -> Path:
    """Validate a target without creating the file or its output directory."""
    _validate_filename(name)
    if not isinstance(directory, str):
        raise ValueError("Choose project, desktop, documents, or downloads as the directory.")
    folder = _folder_path(directory).resolve(strict=directory != "project")
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"The {directory} location is not a directory: {folder}")
    target = folder / name
    # Resolve the leaf only to detect redirections; return the original path so
    # exclusive creation refuses an existing link instead of following it.
    if target.resolve().parent != folder:
        raise ValueError("The filename redirects outside the selected directory; choose a new filename.")
    return target


def _identity(info: os.stat_result) -> dict:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "size": info.st_size,
    }


def _is_redirect(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def create_empty_file(name: str, directory: str = "project") -> dict:
    """Create a new file, returning enough identity data to undo only that file."""
    try:
        target = resolve_file_target(name, directory)
        if directory == "project":
            # Revalidate after creation so an existing redirection cannot send
            # generated data outside the project's ignored output folder.
            created_folder_path(create=True)
            target = resolve_file_target(name, directory)
        # 'x' is an atomic exclusive create: an existing file, directory, or
        # dangling symlink causes failure without changing its contents.
        with target.open("x", encoding="utf-8"):
            pass
        info = target.lstat()
        return {
            "ok": True,
            "path": str(target),
            "created": True,
            "bytes": 0,
            "undo": {"path": str(target), "parent": str(target.parent),
                     "identity": _identity(info)},
        }
    except FileExistsError:
        return {"error": f"{name!r} already exists in {directory}. Choose a different filename; nothing was overwritten."}
    except FileNotFoundError:
        return {"error": f"Your {directory} directory does not exist or is unavailable. Open it in your file manager and try again."}
    except (OSError, ValueError, RuntimeError) as exc:
        return {"error": f"Could not create the file: {exc}"}


def undo_created_file(payload: dict) -> dict:
    """Delete only the unchanged empty file recorded by create_empty_file.

    Raise on refusal so the journal leaves the action available for review,
    instead of recording a refused deletion as a successful undo.
    """
    try:
        target = Path(payload["path"])
        parent = Path(payload["parent"])
        expected = payload["identity"]
        if not target.is_absolute() or target.parent != parent:
            raise ValueError("invalid saved path")
        if parent.resolve(strict=True) != parent or target.resolve().parent != parent:
            raise ValueError("the file's directory has been redirected")
        info = target.lstat()
        if _is_redirect(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("the created file has been replaced")
        if info.st_size != 0 or _identity(info) != expected:
            raise ValueError("the file was edited or replaced; it has been kept")
    except (KeyError, TypeError) as exc:
        raise ValueError("Cannot undo file creation: missing file identity.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Cannot undo file creation: {exc}") from exc
    target.unlink()
    return {"ok": True, "path": str(target), "deleted": True}
