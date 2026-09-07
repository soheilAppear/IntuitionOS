"""The execution environment used by command discovery and gated execution.

The parent terminal is not necessarily our shell: Windows shell=True uses CMD.
PowerShell is an explicit option. Its metadata query and actual invocation use
the same NoProfile session setup, optionally importing a user-created snapshot
of aliases/functions. Discovery never invokes a candidate command.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

_cache = {}
_lock = threading.RLock()
_NAME = re.compile(r"^[%~\w?.:/\\-]+$")


def active_shell_name():
    shell = os.environ.get("INTUITION_SHELL", "cmd" if os.name == "nt" else "sh").lower()
    if shell not in ("cmd", "sh", "powershell", "pwsh"):
        raise ValueError("INTUITION_SHELL must be cmd, sh, powershell, or pwsh")
    return shell


def shell_executable(shell=None):
    shell = shell or active_shell_name()
    if shell == "cmd":
        executable = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
    elif shell == "sh":
        executable = shutil.which("sh") or ("/bin/sh" if Path("/bin/sh").is_file() else None)
    else:
        executable = shutil.which(shell)
    if not executable:
        raise ValueError(f"The selected shell {shell!r} is not installed/on PATH")
    return executable


def load_shell_catalog(shell=None):
    """Read names/definitions, never execute them. Only PS snapshots are supported."""
    shell = shell or active_shell_name()
    path = os.environ.get("INTUITION_SHELL_CATALOG")
    if not path:
        return []
    if shell not in ("powershell", "pwsh"):
        raise ValueError("Shell catalogs require PowerShell; CMD/sh do not inherit functions")
    if Path(path).stat().st_size > 8_000_000:
        raise ValueError("Shell catalog exceeds 8 MB")
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("shell") != shell:
        raise ValueError("Shell catalog belongs to a different execution shell")
    entries = data.get("commands", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or len(entries) > 10000:
        raise ValueError("Invalid shell catalog commands")
    for entry in entries:
        if (not isinstance(entry, dict) or not _NAME.fullmatch(str(entry.get("name", "")))
                or entry.get("kind") not in ("Alias", "Function")
                or not isinstance(entry.get("definition"), str)):
            raise ValueError("Catalog entries must describe named PowerShell aliases/functions")
    return entries


_PS_SETUP = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
if ($env:INTUITION_SHELL_CATALOG) {
    $catalog = Microsoft.PowerShell.Management\Get-Content -Raw -LiteralPath $env:INTUITION_SHELL_CATALOG | Microsoft.PowerShell.Utility\ConvertFrom-Json
    foreach ($entry in $catalog.commands) {
        if ($entry.kind -eq 'Alias') {
            $existing = Microsoft.PowerShell.Management\Get-Item -LiteralPath ('Alias:\' + $entry.name) -ErrorAction SilentlyContinue
            if (-not $existing -or $existing.Definition -ne $entry.definition) {
                if ($existing) {
                    Microsoft.PowerShell.Utility\Set-Alias -Name $entry.name -Value $entry.definition -Scope Global -Force -Option $existing.Options
                } else {
                    Microsoft.PowerShell.Utility\Set-Alias -Name $entry.name -Value $entry.definition -Scope Global -Force
                }
            }
        } elseif ($entry.kind -eq 'Function') {
            Microsoft.PowerShell.Management\Set-Item -LiteralPath ('Function:\global:' + $entry.name) -Value ([scriptblock]::Create($entry.definition))
        }
    }
}
"""


def _ps_argv(script, shell):
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return [shell_executable(shell), "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


def discover_powershell_commands(shell=None):
    """Cold metadata scan; cached for the lifetime of this shell/catalog/PATH."""
    shell = shell or active_shell_name()
    if shell not in ("powershell", "pwsh"):
        return []
    load_shell_catalog(shell)  # validate before PowerShell loads definitions
    path = os.environ.get("INTUITION_SHELL_CATALOG", "")
    stamp = Path(path).stat().st_mtime_ns if path else 0
    key = (shell, path, stamp, os.environ.get("PATH", ""))
    with _lock:
        if key in _cache:
            return list(_cache[key])
        script = _PS_SETUP + r"""
$metadata = @(Microsoft.PowerShell.Core\Get-Command -All | Microsoft.PowerShell.Core\ForEach-Object {
    [pscustomobject]@{name=$_.Name; kind=[string]$_.CommandType; source=$_.Source}
})
[Console]::WriteLine((Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $metadata -Compress))
"""
        result = subprocess.run(_ps_argv(script, shell), capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise ValueError("PowerShell command discovery failed: " + result.stderr[:300])
        entries = json.loads(result.stdout or "[]")
        if isinstance(entries, dict):
            entries = [entries]
        result_names = [e for e in entries if e.get("name")]
        _cache.clear()
        _cache[key] = result_names
        return list(result_names)


def execute_command(cmd, cwd=".", timeout=60):
    """Called only by the capability action, after authorization; no rewriting."""
    shell = active_shell_name()
    load_shell_catalog(shell)
    env = os.environ.copy()
    if shell in ("powershell", "pwsh"):
        # Put user text in an environment value, never interpolate it into setup.
        # The scriptblock runs only here, after run_command's confirmation gate.
        env["INTUITION_SUBMITTED_COMMAND"] = cmd
        argv = _ps_argv(_PS_SETUP + r"""
$global:LASTEXITCODE = 0
try {
    & ([scriptblock]::Create($env:INTUITION_SUBMITTED_COMMAND))
    if (-not $?) { exit 1 }
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
""", shell)
    elif shell == "cmd":
        # A raw command line avoids subprocess.list2cmdline escaping the quotes
        # inside the /c payload. /s strips only this enclosing quote pair.
        argv = f'"{shell_executable(shell)}" /d /s /c "{cmd}"'
    else:
        argv = [shell_executable(shell), "-c", cmd]
    decoding = {"encoding": "utf-8", "errors": "replace"} if shell in ("powershell", "pwsh") else {}
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                            timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **decoding)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
