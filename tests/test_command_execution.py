"""Execution binding and actual shell argument round trips, without OS actions."""
import json
import os
import shutil
import sys

import pytest

from core import actions as actions_mod
from core import shell_environment
from core.capabilities import set_safe_mode


def test_corrected_command_keeps_gate_and_exact_arguments(project, wired, monkeypatch):
    acts, journal, _ = wired
    calls = []
    monkeypatch.setattr(shell_environment, "execute_command",
                        lambda cmd, cwd: calls.append((cmd, cwd)) or {"returncode": 0})
    text = '  python  "a script.py"\t--key="a b"  --lr 0.001  '
    set_safe_mode(True)
    assert acts.call("run_command", cmd=text).get("denied")
    assert not calls
    set_safe_mode(False)
    parked = acts.call("run_command", cmd=text)
    assert parked["needs_confirmation"]
    assert parked["args"]["cmd"] == text
    assert not calls
    assert acts.confirm(parked["token"], True)["returncode"] == 0
    assert calls == [(text, str(project))]
    assert journal.recent()[0]["capability"] == "run_command"
    assert "error" in acts.confirm(parked["token"], True)


def test_reenabling_safe_mode_invalidates_approval(project, wired, monkeypatch):
    acts, _, _ = wired
    calls = []
    monkeypatch.setattr(shell_environment, "execute_command", lambda *a, **k: calls.append(a))
    set_safe_mode(False)
    token = acts.call("run_command", cmd="echo allowed?")["token"]
    set_safe_mode(True)
    assert acts.confirm(token, True)["denied"]
    assert not calls


def test_nonzero_command_exit_is_journalled_as_error(project, wired, monkeypatch):
    acts, journal, _ = wired
    monkeypatch.setattr(shell_environment, "execute_command",
                        lambda *a, **k: {"returncode": 7, "stdout": "", "stderr": "failed"})
    set_safe_mode(False)
    token = acts.call("run_command", cmd="failing-command")["token"]
    assert acts.confirm(token)["returncode"] == 7
    assert journal.recent()[0]["outcome"] == "error"


@pytest.mark.parametrize("tail", [
    '  "a script.py" --label "two words"  --lr 0.001',
    "\ttrain.py --secret='a b'  ",
    ' -c "print(1 + 2)"',
])
def test_legacy_exec_python_replacement_preserves_argument_bytes(tail):
    text = "  python" + tail
    rewritten = actions_mod._swap_interpreter(text, "chosen-python")
    assert rewritten == "  chosen-python" + tail


@pytest.mark.parametrize("shell", ["cmd", "powershell", "pwsh", "sh"])
def test_real_shell_preserves_quoted_python_arguments(shell, project, monkeypatch):
    if shell == "cmd" and os.name != "nt":
        pytest.skip("CMD is Windows only")
    if not shutil.which(shell):
        pytest.skip(f"{shell} not installed")
    monkeypatch.setenv("INTUITION_SHELL", shell)
    monkeypatch.delenv("INTUITION_SHELL_CATALOG", raising=False)
    script = project / "argument probe.py"
    script.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    prefix = "& " if shell in ("powershell", "pwsh") else ""
    cmd = f'{prefix}"{sys.executable}"  "{script}"  "two words" "" --lr 0.001 "C:\\a b\\file.txt"'
    result = shell_environment.execute_command(cmd, str(project))
    assert result["returncode"] == 0, result
    # Windows PowerShell 5.1 uses the old native-argument binder, which drops
    # empty native arguments; the resolver must not claim to fix shell semantics.
    expected = ["two words", "", "--lr", "0.001", "C:\\a b\\file.txt"]
    if shell == "powershell":
        expected.remove("")
    assert json.loads(result["stdout"].strip()) == expected


@pytest.mark.parametrize("shell", ["powershell", "pwsh"])
def test_discovery_and_execution_share_alias_function_snapshot(shell, project, monkeypatch):
    if not shutil.which(shell):
        pytest.skip(f"{shell} not installed")
    marker = project / "must-not-be-created.txt"
    catalog = project / "shell.json"
    catalog.write_text(json.dumps({"shell": shell, "commands": [
        {"name": "Invoke-IntuitionProbe", "kind": "Function", "definition": "param($value) Write-Output $value"},
        {"name": "iprobe", "kind": "Alias", "definition": "Invoke-IntuitionProbe"},
        {"name": "Never-InvokeMe", "kind": "Function", "definition": f"Set-Content -LiteralPath '{marker}' -Value bad"},
    ]}), encoding="utf-8")
    monkeypatch.setenv("INTUITION_SHELL", shell)
    monkeypatch.setenv("INTUITION_SHELL_CATALOG", str(catalog))
    names = {item["name"] for item in shell_environment.discover_powershell_commands()}
    assert {"Invoke-IntuitionProbe", "iprobe", "Never-InvokeMe"} <= names
    assert not marker.exists(), "Discovery executed a candidate function"
    result = shell_environment.execute_command('iprobe "hello from the alias"', str(project))
    assert result["returncode"] == 0, result
    assert result["stdout"].strip() == "hello from the alias"
    assert not marker.exists()


def test_catalog_cannot_claim_another_execution_environment(project, monkeypatch):
    catalog = project / "shell.json"
    catalog.write_text(json.dumps({"shell": "cmd", "commands": []}), encoding="utf-8")
    monkeypatch.setenv("INTUITION_SHELL_CATALOG", str(catalog))
    with pytest.raises(ValueError, match="different execution shell"):
        shell_environment.load_shell_catalog("pwsh")
    with pytest.raises(ValueError, match="require PowerShell"):
        shell_environment.load_shell_catalog("cmd")


@pytest.mark.parametrize("shell", ["powershell", "pwsh"])
def test_discovery_never_calls_shadowed_metadata_functions(shell, project, monkeypatch):
    if not shutil.which(shell):
        pytest.skip(f"{shell} not installed")
    marker = project / "candidate-was-executed.txt"
    names = ['Get-Command', 'ForEach-Object', 'ConvertTo-Json', 'Set-Item',
             'Get-Alias', 'Set-Alias', 'Get-Content', 'ConvertFrom-Json', 'Out-Default']
    catalog = project / "shadowed.json"
    commands = [{"name": name, "kind": "Function",
                 "definition": f"[IO.File]::WriteAllText('{marker}', 'bad'); throw 'Candidate executed'"}
                for name in names]
    commands.extend([{"name": "%", "kind": "Alias", "definition": "ForEach-Object"},
                     {"name": "cd~", "kind": "Function", "definition": "Set-Location ~"}])
    catalog.write_text(json.dumps({"shell": shell, "commands": commands}), encoding="utf-8")
    monkeypatch.setenv("INTUITION_SHELL", shell)
    monkeypatch.setenv("INTUITION_SHELL_CATALOG", str(catalog))
    discovered = shell_environment.discover_powershell_commands()
    assert 'Get-Command' in {item['name'] for item in discovered}
    assert not marker.exists(), "Metadata discovery executed a candidate function"
