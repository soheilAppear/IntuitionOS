"""Shared fixtures.

Two things need care across the suite. Capability state is module-level, so a
test that flips Safe Mode must put it back. And the project path jail is anchored
to os.getcwd(), so a test about paths has to control the working directory rather
than inherit whatever launched pytest.
"""

import os

import pytest

from core import actions as actions_mod
from core.capabilities import capabilities, set_safe_mode
from core.journal import Journal
from core.memory import Memory


@pytest.fixture(autouse=True)
def restore_safe_mode():
    # Safe Mode is process-wide; leave it as we found it.
    before = actions_mod._is_safe()
    yield
    set_safe_mode(before)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway directory standing in as the project root."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def memory(tmp_path):
    return Memory(str(tmp_path / "db" / "test.db"))


@pytest.fixture
def wired(memory, monkeypatch):
    """actions wired to a fresh database, with the journal open on it."""
    monkeypatch.setattr(actions_mod, "_memory", memory)
    journal = Journal(memory)
    monkeypatch.setattr(actions_mod, "_journal_ref", [journal])
    monkeypatch.setattr(actions_mod, "_logger", lambda _msg: None)
    return actions_mod.actions, journal, memory


@pytest.fixture
def registry():
    return capabilities
