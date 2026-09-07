"""Cold PowerShell discovery must tolerate real module catalogs and cache them."""

import subprocess
from unittest.mock import Mock

import pytest

from core import shell_environment as shell


@pytest.fixture(autouse=True)
def empty_catalog(monkeypatch):
    monkeypatch.delenv("INTUITION_SHELL_CATALOG", raising=False)
    monkeypatch.setattr(shell, "_cache", {})
    monkeypatch.setattr(shell, "shell_executable", lambda name=None: "powershell")


def test_cold_discovery_has_larger_bounded_budget_and_reuses_result(monkeypatch):
    run = Mock(return_value=Mock(returncode=0, stdout='[{"name":"Get-Item"}]'))
    monkeypatch.setattr(shell.subprocess, "run", run)
    assert shell.discover_powershell_commands("powershell") == [{"name": "Get-Item"}]
    shell.discover_powershell_commands("powershell")
    assert run.call_count == 1
    assert 60 <= run.call_args.kwargs["timeout"] <= 120


def test_module_path_change_invalidates_discovery_cache(monkeypatch):
    run = Mock(return_value=Mock(returncode=0, stdout="[]"))
    monkeypatch.setattr(shell.subprocess, "run", run)
    monkeypatch.setenv("PSModulePath", "first")
    shell.discover_powershell_commands("powershell")
    monkeypatch.setenv("PSModulePath", "second")
    shell.discover_powershell_commands("powershell")
    assert run.call_count == 2


def test_discovery_timeout_explains_setup_failure_without_encoded_command(monkeypatch):
    monkeypatch.setattr(shell.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("encoded payload", 120)))
    with pytest.raises(ValueError, match="PowerShell command discovery timed out") as error:
        shell.discover_powershell_commands("powershell")
    assert "encoded payload" not in str(error.value)


@pytest.mark.parametrize("body", ["[]", "null", "42", '"text"'])
def test_catalog_requires_json_object(tmp_path, monkeypatch, body):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(body, encoding="utf-8")
    monkeypatch.setenv("INTUITION_SHELL_CATALOG", str(catalog))
    with pytest.raises(ValueError, match="object"):
        shell.load_shell_catalog("powershell")
