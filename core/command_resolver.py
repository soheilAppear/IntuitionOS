"""Deterministic command suggestions, argument-safe edits and explicit feedback.

Providers describe commands in the execution shell; metadata discovery never
invokes a proposed candidate. CommandResolver selects a safe span and ranks
replacements. CorrectionSession binds a displayed result to one submission.
The interfaces still send the committed text through the capability gate.

Scores are ranking weights, not probabilities. Persistent feedback contains
command tokens only. Full input and candidate text remain transient in a session.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol

from .os_intents import is_app_command


KNOWN_COMMANDS = [
    "/help",
    "/exit",
    "/memory",
    "/dream",
    "/save",
    "/recall",
    "/actions",
    "/config",
    "/reload",
    "/tasks",
    "/done",
    "/delete",
    "/snooze",
    "/hw",
    "/task_payload",
    "/safe",
    "/exec",
    "/write",
    "/read",
    "/undo",
    "/journal",
    "/capabilities",
    "/forget",
    "/episodes",
    "/calibration",
    "/thresholds",
    "/rules",
]


def _value(context, key, default=None):
    """Accept either a Context snapshot or the equivalent mapping."""
    if isinstance(context, dict):
        return context.get(key, default)
    return getattr(context, key, default)


def _project(context):
    """Return a stable project key without persisting its filesystem path."""
    cwd = _value(context, "cwd", "") or ""
    return (
        hashlib.sha256(os.path.normcase(os.path.abspath(cwd)).encode()).hexdigest()
        if cwd
        else ""
    )


@dataclass(frozen=True)
class Command:
    """One available name, its routing namespace, kind and shell case rules."""

    name: str
    namespace: str = "shell"
    kind: str = "executable"
    case_sensitive: bool = True


class CandidateProvider(Protocol):
    """Extensible command catalog; implementations may perform metadata reads."""

    def commands(self, context=None) -> Iterable[Command]:
        """Yield commands available in this provider's execution environment."""
        ...


class StaticCommandProvider:
    """Injectable vocabulary; aliases/functions require an execution catalog."""

    def __init__(
        self, names, namespace="shell", case_sensitive=True, kind="executable"
    ):
        self.entries = tuple(
            n if isinstance(n, Command) else Command(n, namespace, kind, case_sensitive)
            for n in names
        )

    def commands(self, context=None):
        return self.entries


class IntuitionCommandProvider(StaticCommandProvider):
    """Slash commands and the built-in directory-listing forms."""

    def __init__(self):
        super().__init__(
            KNOWN_COMMANDS + ["ls", "tree"],
            namespace="intuitionos",
            case_sensitive=True,
            kind="internal",
        )


class ShellBuiltinProvider(StaticCommandProvider):
    """Static built-ins for CMD/sh; PowerShell supplies its own metadata."""

    CMD = (
        "assoc break call cd chdir cls color copy date del dir echo endlocal erase "
        "exit for ftype goto if md mkdir mklink move path pause popd prompt pushd "
        "rd rem ren rename rmdir set setlocal shift start time title type ver verify vol"
    ).split()
    SH = (
        ": . [ alias bg break cd command continue eval exec exit export false fc fg "
        "getopts hash jobs kill printf pwd read readonly return set shift test times "
        "trap true type ulimit umask unalias unset wait"
    ).split()

    def __init__(self, shell="sh"):
        super().__init__(
            self.CMD if shell == "cmd" else self.SH if shell == "sh" else [],
            case_sensitive=shell == "sh",
            kind="builtin",
        )


class EnvironmentCatalogProvider(StaticCommandProvider):
    """Only expose a catalog bound to the shell that will actually execute it.

    Parent interactive shell state is not inherited by a fresh CMD/sh process.
    The caller must install catalog aliases/functions into its execution shell.
    """

    def __init__(self, commands, *, shell, supported_shell):
        entries = []
        if shell == supported_shell:
            for item in commands:
                if isinstance(item, Command):
                    entries.append(item)
                elif isinstance(item, str):
                    entries.append(Command(item, case_sensitive=shell == "sh"))
                else:
                    entries.append(
                        Command(
                            item["name"],
                            "shell",
                            item.get("kind", "command"),
                            shell == "sh",
                        )
                    )
                    source = item.get("source")
                    if (
                        shell in {"powershell", "pwsh"}
                        and source
                        and item.get("kind", "").lower()
                        in {"alias", "function", "cmdlet"}
                    ):
                        entries.append(
                            Command(
                                source + "\\" + item["name"],
                                "shell",
                                item.get("kind", "command"),
                                False,
                            )
                        )
        super().__init__(entries)


class PathExecutableProvider:
    """Filesystem-only PATH discovery, cached briefly and refreshed on PATH/cwd.

    On Windows PATHEXT entries are discoverable scripts/executables. On POSIX
    only executable regular files qualify; a .py file alone is not a command.
    The three-second cache expires on the next read and also keys on PATH,
    PATHEXT, cwd and shell. Earlier PATH entries win duplicate-name resolution.
    """

    def __init__(self, *, shell="sh", environ=None, ttl=3.0):
        self.shell, self.environ, self.ttl = (
            shell,
            os.environ if environ is None else environ,
            ttl,
        )
        self._key, self._at, self._entries = None, 0.0, ()

    def commands(self, context=None):
        env = self.environ
        cwd = _value(context, "cwd", None) or os.getcwd()
        windows = self.shell in {"cmd", "powershell", "pwsh"} and os.name == "nt"
        path = env.get("PATH", "")
        pathext = env.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
        key = (path, pathext, cwd, self.shell)
        now = time.monotonic()
        if key == self._key and now - self._at < self.ttl:
            return self._entries
        extensions = {e.lower() for e in pathext.split(";") if e}
        directories = path.split(os.pathsep)
        if self.shell == "cmd":
            directories.insert(0, cwd)
        found = {}
        for directory in directories:
            directory = directory.strip('"') or cwd
            if not os.path.isabs(directory):
                directory = os.path.join(cwd, directory)
            try:
                with os.scandir(directory) as items:
                    for item in items:
                        try:
                            if not item.is_file():
                                continue
                            suffix = Path(item.name).suffix.lower()
                            if windows and suffix not in extensions:
                                continue
                            if not windows and not os.access(item.path, os.X_OK):
                                continue
                            names = [item.name]
                            if windows:
                                names.append(item.name[: -len(suffix)])
                            for name in names:
                                key_name = (
                                    name.casefold() if self.shell != "sh" else name
                                )
                                found.setdefault(
                                    key_name,
                                    Command(
                                        name,
                                        case_sensitive=self.shell == "sh",
                                        kind=(
                                            "script"
                                            if suffix in {".cmd", ".bat", ".ps1"}
                                            else "executable"
                                        ),
                                    ),
                                )
                        except OSError:
                            continue
            except OSError:
                continue
        self._key, self._at = key, now
        self._entries = tuple(found.values())
        return self._entries


class GitSubcommandProvider:
    """Grammar: `git SUBCOMMAND [opaque arguments]` only, plus explicit aliases.

    Leading git options are left untouched: safely understanding -C/-c and
    their option arguments needs a larger grammar. No arbitrary argument edits.
    This injectable catalog reads basic alias config. The default runtime uses
    InstalledGitSubcommandProvider so Git resolves its complete configuration.
    """

    DEFAULT = (
        "add am annotate apply archive bisect blame branch bundle cat-file "
        "check-attr check-ignore check-mailmap check-ref-format checkout cherry "
        "cherry-pick clean clone column commit config count-objects credential "
        "describe diff difftool fast-export fast-import fetch filter-branch "
        "for-each-ref format-patch fsck gc grep help init log ls-files ls-remote "
        "ls-tree maintenance merge merge-base mergetool mv notes pack-objects "
        "prune pull push range-diff read-tree rebase reflog remote repack replace "
        "request-pull reset restore revert rev-list rev-parse rm shortlog show "
        "show-branch sparse-checkout stash status submodule switch symbolic-ref "
        "tag update-index update-ref verify-commit verify-tag version whatchanged "
        "worktree write-tree"
    ).split()

    def __init__(self, names=None, aliases=()):
        self.names = tuple(self.DEFAULT if names is None else names)
        self.aliases = tuple(aliases)

    def commands(self, context=None):
        names = set(self.names) | set(self.aliases)
        cwd = Path(_value(context, "cwd", None) or os.getcwd())
        configs = [
            Path.home() / ".gitconfig",
            Path.home() / ".config" / "git" / "config",
        ]
        for parent in (cwd, *cwd.parents):
            if (parent / ".git").is_dir():
                configs.append(parent / ".git" / "config")
                break
        # Config reads only: do not invoke git or execute alias definitions.
        for config in configs:
            parser = configparser.RawConfigParser(strict=False)
            try:
                parser.read(config, encoding="utf-8")
                if parser.has_section("alias"):
                    names.update(parser.options("alias"))
            except (OSError, configparser.Error, UnicodeError):
                pass
        return tuple(Command(name, "git", "subcommand", True) for name in sorted(names))


class InstalledGitSubcommandProvider:
    """Fixed metadata query, never execute a suggested Git subcommand.

    Git itself resolves included/conditional/worktree config and aliases. If
    metadata cannot be read, subcommand correction is disabled rather than
    assuming an unknown-but-valid Git alias is a misspelling.
    Results, including failures, are cached for thirty seconds per executable,
    cwd and Git configuration environment. The query has a one-second timeout.
    """

    def __init__(self, ttl=30.0):
        self.ttl = ttl
        self._cache = {}
        self.available = False

    def commands(self, context=None):
        cwd = _value(context, "cwd", None) or os.getcwd()
        executable = shutil.which("git")
        if not executable:
            self.available = False
            return ()
        key = (
            executable,
            cwd,
            os.environ.get("GIT_CONFIG_GLOBAL"),
            os.environ.get("GIT_CONFIG_SYSTEM"),
            os.environ.get("GIT_DIR"),
        )
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.ttl:
            self.available = cached[2]
            return cached[1]
        names, available = (), False
        try:
            result = subprocess.run(
                [executable, "--list-cmds=main,others,alias"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=1.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                names = tuple(
                    Command(name, "git", "subcommand", True)
                    for name in sorted(set(result.stdout.split()))
                    if re.fullmatch(r"[A-Za-z0-9_.+-]+", name)
                )
                available = bool(names)
        except (OSError, subprocess.SubprocessError, UnicodeError):
            pass
        self.available = available
        self._cache[key] = (now, names, available)
        return names


@dataclass(frozen=True)
class Candidate:
    """One full replacement line, with its changed token and ranking evidence.

    ``span`` is a half-open pair of Python string offsets into the original
    input. Replacing that slice with ``token`` produces ``text``; all surrounding
    characters stay identical. ``score`` is a deterministic weight, not a probability.
    """

    text: str
    token: str
    namespace: str
    score: float
    reason: str
    span: tuple[int, int]

    def to_dict(self):
        return asdict(self)


@dataclass
class Resolution:
    """Original input plus exact/incomplete/correction/ambiguous/unsupported state.

    Candidates are ordered best first. ``namespace`` identifies IntuitionOS,
    shell or Git routing; ``unknown`` leaves ordinary language to the interface.
    An exact result preserves a known command head, not validation of its args.
    """

    original: str
    status: str
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""
    span: tuple[int, int] | None = None
    namespace: str = "unknown"

    def to_dict(self):
        return asdict(self)


def spelling_distance(left: str, right: str) -> int:
    """Optimal-string-alignment distance including adjacent transpositions."""
    rows = [list(range(len(right) + 1))]
    for i, a in enumerate(left, 1):
        row = [i] + [0] * len(right)
        for j, b in enumerate(right, 1):
            row[j] = min(
                row[j - 1] + 1, rows[i - 1][j] + 1, rows[i - 1][j - 1] + (a != b)
            )
            if i > 1 and j > 1 and a == right[j - 2] and left[i - 2] == b:
                row[j] = min(row[j], rows[i - 2][j - 2] + 1)
        rows.append(row)
    return rows[-1][-1]


def _tokens(text, shell):
    """Return half-open token spans, or None when shell syntax is unsupported.

    Quotes remain in the original text. This recognizes enough grammar to edit
    a command name safely; it is deliberately not a general shell parser.
    """
    spans, start, quote = [], None, None
    for i, char in enumerate(text):
        if char in "\r\n\x00":
            return None
        if quote:
            if char == quote:
                quote = None
            # Expansion/escaping inside double quotes has shell-specific rules.
            elif (
                quote == '"' and (char in "$`" or (char == "%" and shell == "cmd"))
            ) or (char == "\\" and shell == "sh"):
                return None
            continue
        if char == '"' or (char == "'" and shell != "cmd"):
            if start is None:
                start = i
            quote = char
        elif (
            char in "|&;<>`(){}$!^"
            or (char == "%" and shell == "cmd")
            or (char == "\\" and shell == "sh")
        ):
            return None
        elif char.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        else:
            if start is None:
                start = i
    if quote:
        return None
    if start is not None:
        spans.append((start, len(text)))
    return spans


class CommandResolver:
    """Rank available command names without submitting or executing user input.

    Providers are ordered: the first exact entry determines command kind, which
    matters when an alias shadows an executable. ``context`` may be a Context
    or dict; cwd selects project evidence and ts bounds chronological learning.
    """

    def __init__(
        self,
        providers=None,
        *,
        shell="sh",
        feedback=None,
        git_provider=None,
        max_candidates=3,
    ):
        self.shell = shell
        self.providers = list(
            providers
            if providers is not None
            else [
                IntuitionCommandProvider(),
                ShellBuiltinProvider(shell),
                PathExecutableProvider(shell=shell),
            ]
        )
        self.feedback = feedback
        self.git_provider = git_provider or GitSubcommandProvider()
        self.max_candidates = max_candidates

    @property
    def generation(self):
        return self.feedback.generation if self.feedback else 0

    def resolve(self, text, context=None):
        """Return a Resolution without changing text or recording acceptance.

        Existing app intents take precedence for bare input. ``/exec`` explicitly
        chooses shell grammar. Only the command name, or Git's immediate
        subcommand when supported, can become a replacement span.
        """
        original = text
        if not text.strip():
            return Resolution(original, "incomplete", reason="Enter a command.")
        first = re.search(r"\S+", text)
        start, end = first.span()
        token = first.group()
        if not token.startswith("/") and is_app_command(original):
            return Resolution(
                original,
                "exact",
                reason="Existing IntuitionOS intent; input is preserved.",
                span=(start, end),
                namespace="intuitionos",
            )
        if token in {"ls", "tree"}:
            return Resolution(
                original,
                "exact",
                reason="IntuitionOS command; input is preserved.",
                span=(start, end),
                namespace="intuitionos",
            )
        # Internal command arguments are opaque data, not parsed as shell text.
        if token.startswith("/") and not (self.shell == "sh" and os.path.isfile(token)):
            if token != "/exec":
                commands = [
                    c
                    for p in self.providers
                    for c in p.commands(context)
                    if c.namespace == "intuitionos"
                ]
                return self._rank(
                    original, (start, end), commands, "intuitionos", context
                )
            body = text[end:]
            if not body.strip():
                return Resolution(
                    original,
                    "incomplete",
                    reason="Enter a shell command after /exec.",
                    namespace="shell",
                )
            spans = _tokens(body, self.shell)
            if spans is not None:
                spans = [(a + end, b + end) for a, b in spans]
        else:
            spans = _tokens(text, self.shell)
        if spans is None:
            return Resolution(
                original,
                "unsupported",
                reason=(
                    "Shell operators, expansions, escapes or unmatched quotes "
                    "prevent a safe correction span; input is unchanged."
                ),
                namespace="shell",
            )
        span = spans[0]
        head = original[span[0] : span[1]]
        if any(c in head for c in "'\"="):
            return Resolution(
                original,
                "unsupported",
                reason="Quoted command names and environment assignments are not corrected.",
                span=span,
                namespace="shell",
            )
        commands = [
            c
            for p in self.providers
            for c in p.commands(context)
            if c.namespace != "intuitionos"
        ]
        exact = next((c for c in commands if self._equal(head, c)), None)
        if "/" in head or "\\" in head:
            if exact:
                return Resolution(
                    original,
                    "exact",
                    reason="Available qualified command; input is preserved.",
                    span=span,
                    namespace="shell",
                )
            if self.shell in {"powershell", "pwsh"} and re.fullmatch(
                r"[A-Za-z0-9_.-]+\\[^\\/]+", head
            ):
                module = head.split("\\", 1)[0].casefold() + "\\"
                qualified = [
                    c
                    for c in commands
                    if c.name.casefold().startswith(module)
                    and c.kind.lower() in {"alias", "function", "cmdlet"}
                ]
                if qualified:
                    return self._rank(original, span, qualified, "shell", context)
            path = Path(head)
            if not path.is_absolute():
                path = Path(_value(context, "cwd", None) or os.getcwd()) / path
            extensions = {
                e.lower()
                for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
            }
            if self.shell in {"powershell", "pwsh"}:
                extensions.add(".ps1")
            available = path.is_file() and (
                os.access(path, os.X_OK)
                if self.shell == "sh"
                else path.suffix.lower() in extensions
            )
            return Resolution(
                original,
                "exact" if available else "unsupported",
                reason=(
                    "Explicit executable path is preserved."
                    if available
                    else "Executable path is unavailable; paths are not corrected."
                ),
                span=span,
                namespace="shell",
            )
        if (
            exact
            and exact.kind.lower() in {"executable", "application"}
            and head.casefold() in {"git", "git.exe"}
            and len(spans) > 1
        ):
            subspan = spans[1]
            subcommand = original[subspan[0] : subspan[1]]
            if subcommand.startswith("-"):
                return Resolution(
                    original,
                    "exact",
                    reason="Known git command; options and arguments remain untouched.",
                    span=span,
                    namespace="shell",
                )
            if any(c in subcommand for c in "'\"/\\="):
                return Resolution(
                    original,
                    "unsupported",
                    reason="Git subcommand grammar is not established.",
                    span=subspan,
                    namespace="git",
                )
            git_commands = list(self.git_provider.commands(context))
            if (
                isinstance(self.git_provider, InstalledGitSubcommandProvider)
                and not self.git_provider.available
            ):
                return Resolution(
                    original,
                    "exact",
                    reason=(
                        "Git is available; subcommand metadata is unavailable, "
                        "so arguments remain untouched."
                    ),
                    span=span,
                    namespace="shell",
                )
            git_commands.extend(
                Command(c.name[4:], "git", "external-subcommand", True)
                for c in commands
                if c.name.startswith("git-")
            )
            return self._rank(original, subspan, git_commands, "git", context)
        return self._rank(original, span, commands, "shell", context)

    @staticmethod
    def _equal(token, command):
        return (
            token == command.name
            if command.case_sensitive
            else token.casefold() == command.name.casefold()
        )

    def _rank(self, original, span, commands, namespace, context):
        """Preserve exact names, then rank plausible edits within this namespace."""
        token = original[span[0] : span[1]]
        if any(self._equal(token, c) for c in commands):
            return Resolution(
                original,
                "exact",
                reason="Available command; input is preserved.",
                span=span,
                namespace=namespace,
            )
        if len(token.lstrip("/")) < 2:
            return Resolution(
                original,
                "incomplete" if token == "/" else "unsupported",
                reason="Too little command text for a safe suggestion.",
                span=span,
                namespace=namespace if namespace != "shell" else "unknown",
            )
        candidates, seen = [], set()
        for command in commands:
            name = command.name
            key = name if command.case_sensitive else name.casefold()
            if key in seen:
                continue
            seen.add(key)
            typed = token if command.case_sensitive else token.casefold()
            wanted = name if command.case_sensitive else name.casefold()
            # A prefix is an incomplete name, even when deletion distance is high.
            prefix = wanted.startswith(typed)
            max_distance = 1 if len(typed.lstrip("/")) <= 4 else 2
            if not prefix and abs(len(typed) - len(wanted)) > max_distance:
                continue
            distance = spelling_distance(typed, wanted)
            similarity = 1.0 - distance / max(len(typed), len(wanted), 1)
            if not prefix and (distance > max_distance or similarity < 0.55):
                continue
            baseline = (
                (0.68 + 0.22 * len(typed) / len(wanted)) if prefix else similarity
            )
            boost = (
                self.feedback.boost(token, name, namespace, context)
                if self.feedback
                else 0.0
            )
            score = round(baseline + boost, 6)
            reason = (
                "Complete command name"
                if prefix
                else f"Spelling distance {distance} (adjacent transpositions count as one)"
            )
            if boost:
                reason += "; prior explicit acceptance and project context"
            candidates.append(
                Candidate(
                    original[: span[0]] + name + original[span[1] :],
                    name,
                    namespace,
                    score,
                    reason,
                    span,
                )
            )
        candidates.sort(key=lambda c: (-c.score, c.token.casefold(), c.token))
        if not candidates:
            return Resolution(
                original,
                "unsupported",
                reason="No plausible available command; input is unchanged.",
                span=span,
                namespace=namespace if namespace != "shell" else "unknown",
            )
        status = (
            "incomplete"
            if candidates[0].reason.startswith("Complete")
            else "correction"
        )
        if len(candidates) > 1 and candidates[0].score - candidates[1].score < 0.045:
            status = "ambiguous"
        return Resolution(
            original,
            status,
            candidates[: self.max_candidates],
            "Choose a displayed suggestion or keep the original.",
            span,
            namespace,
        )


_FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS command_corrections (
 id INTEGER PRIMARY KEY, ts REAL NOT NULL, namespace TEXT NOT NULL,
 original_token TEXT NOT NULL, candidates_json TEXT NOT NULL, project_key TEXT NOT NULL,
 selected_token TEXT, selected_ts REAL, accepted INTEGER, manual_token TEXT,
 outcome TEXT, outcome_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_command_corrections_ts ON command_corrections(ts);
CREATE TABLE IF NOT EXISTS command_correction_meta (id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL);
INSERT OR IGNORE INTO command_correction_meta(id,generation) VALUES(1,0);
"""


def _safe_token(value):
    # A command token is the only learning unit; never retain arguments/paths.
    value = str(value or "")
    return value[:100] if re.fullmatch(r"/?[A-Za-z0-9_.:+-]+", value) else ""


class CorrectionFeedbackStore:
    """SQLite correction evidence stored through Memory's synchronized wrapper.

    ``enabled`` is a bool or callable, allowing the logging switch to take effect
    without rebuilding the resolver. Only explicit candidate selections produce
    accepted evidence; manual edits, keeping the original and outcomes are
    recorded separately. No full candidate line or argument value is persisted.
    """

    def __init__(self, memory, enabled=True):
        self.mem, self.enabled = memory, enabled
        self.mem.executescript(_FEEDBACK_SCHEMA)

    @property
    def active(self):
        return bool(self.enabled() if callable(self.enabled) else self.enabled)

    @property
    def generation(self):
        """Shared invalidation counter read from SQLite, including other sessions."""
        rows = self.mem.query(
            "SELECT generation FROM command_correction_meta WHERE id=1"
        )
        return rows[0][0] if rows else 0

    def record_display(self, resolution, context=None, ts=None):
        """Return a display-event ID, or None if logging/suggestions are absent.

        Call when presenting candidates, not for speculative background resolves.
        Explicit ts values support replay tests; live calls use wall-clock time.
        """
        if not self.active or not resolution.candidates or resolution.span is None:
            return None
        token = _safe_token(resolution.original[slice(*resolution.span)])
        candidates = [
            {"token": _safe_token(c.token), "namespace": c.namespace, "score": c.score}
            for c in resolution.candidates
        ]
        return self.mem.insert(
            "INSERT INTO command_corrections(ts,namespace,original_token,candidates_json,project_key) VALUES(?,?,?,?,?)",
            (
                time.time() if ts is None else ts,
                resolution.namespace,
                token,
                json.dumps(candidates),
                _project(context),
            ),
        )

    def record_selection(
        self, event_id, candidate_index=None, manual_token=None, ts=None
    ):
        """Record one selected index or an ambiguous keep/manual-edit event.

        A missing index leaves accepted NULL, rather than treating silence or
        an edit as rejection. Invalid indices raise ValueError before a write.
        """
        if not self.active or event_id is None:
            return
        rows = self.mem.query(
            "SELECT candidates_json FROM command_corrections WHERE id=?", (event_id,)
        )
        if not rows:
            return
        choices = json.loads(rows[0][0])
        if candidate_index is not None and (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(choices)
        ):
            raise ValueError("Invalid correction selection")
        selected = (
            choices[candidate_index]["token"] if candidate_index is not None else None
        )
        self.mem.execute(
            "UPDATE command_corrections SET selected_token=?,selected_ts=?,accepted=?,manual_token=? WHERE id=?",
            (
                selected,
                time.time() if ts is None else ts,
                1 if selected else None,
                _safe_token(manual_token),
                event_id,
            ),
        )

    def record_outcome(self, event_id, outcome, ts=None):
        """Attach a result category without changing correction acceptance."""
        if self.active and event_id is not None:
            # Outcomes are categories, never tool result strings with secrets.
            category = (
                outcome
                if outcome
                in {
                    "ok",
                    "error",
                    "denied",
                    "cancelled",
                    "pending",
                    "undone",
                    "unsupported",
                }
                else "unknown"
            )
            self.mem.execute(
                "UPDATE command_corrections SET outcome=?,outcome_ts=? WHERE id=?",
                (category, time.time() if ts is None else ts, event_id),
            )

    def boost(self, original, candidate, namespace, context=None):
        """Return an additive ranking weight in [0, 0.35] from earlier acceptances.

        Display and selection timestamps must be at or before context.ts. Frequency,
        repeated typo mappings, the same project and recency contribute; command
        success does not. Disabling logging also disables learned ranking.
        """
        if not self.active:
            return 0.0
        now = _value(context, "ts", None)
        now = time.time() if now is None else float(now)
        rows = self.mem.query(
            "SELECT original_token,project_key,selected_ts FROM command_corrections WHERE namespace=? AND selected_token=? AND accepted=1 AND ts<=? AND selected_ts<=? ORDER BY selected_ts DESC LIMIT 256",
            (namespace, candidate, now, now),
        )
        if not rows:
            return 0.0
        same = sum(row[0] == original for row in rows)
        project = _project(context)
        local = sum(row[1] == project for row in rows) if project else 0
        recency = math.exp(-max(0, now - rows[0][2]) / (30 * 86400))
        return min(
            0.35,
            0.045 * math.log1p(len(rows))
            + 0.09 * math.log1p(same)
            + 0.08 * math.log1p(local)
            + 0.035 * recency,
        )

    def forget(self, before=None, *, since=None):
        """Forget all history, or rows older than the episode cutoff `before`.

        ``since`` is a compatibility spelling with the same older-than semantics;
        ``before`` takes precedence. Every call advances the shared generation
        so previously displayed corrections cannot be committed after forgetting.
        """
        cutoff = before if before is not None else since
        if cutoff is None:
            self.mem.execute("DELETE FROM command_corrections")
        else:
            self.mem.execute("DELETE FROM command_corrections WHERE ts<?", (cutoff,))
        self.mem.execute(
            "UPDATE command_correction_meta SET generation=generation+1 WHERE id=1"
        )


class CorrectionSession:
    """Bind displayed text to a one-use selection, independently of permission.

    update() prepares a display; snapshot() adds its opaque token and revision.
    commit() checks the original text, token, revision and forget generation
    before returning the chosen full line. It never executes that line.
    invalidate() prevents stale selection while retaining manual-edit evidence.
    """

    def __init__(self, resolver):
        self.resolver = resolver
        self.token = None
        self.revision = 0
        self.resolution = None
        self.feedback_id = None
        self._context = None
        self._generation = resolver.generation
        self._committed = False

    def update(self, text, context=None):
        """Resolve a display revision and return its Resolution.

        Identical text with a live token and unchanged generation reuses the
        display. A changed line records only its edited grammatical token, using
        the previous slot: command head and Git subcommand are distinct.
        """
        generation = self.resolver.generation
        if (
            self.resolution is not None
            and self.resolution.original == text
            and self.token
            and generation == self._generation
        ):
            return self.resolution
        previous = self.resolution
        resolution = self.resolver.resolve(text, context)
        if (
            previous
            and previous.original != text
            and self.resolver.feedback
            and not self._committed
        ):
            # Manual edits are ambiguous evidence, never automatic acceptance.
            spans = _tokens(text, self.resolver.shell) or []
            if spans and text[slice(*spans[0])] == "/exec":
                spans = spans[1:]
            index = 1 if previous.namespace == "git" else 0
            manual = text[slice(*spans[index])] if len(spans) > index else ""
            self.resolver.feedback.record_selection(
                self.feedback_id, manual_token=manual
            )
        self.revision += 1
        self.token = uuid.uuid4().hex
        self.resolution = resolution
        self._context, self._generation, self._committed = context, generation, False
        self.feedback_id = (
            self.resolver.feedback.record_display(self.resolution, context)
            if self.resolver.feedback
            else None
        )
        return self.resolution

    def snapshot(self):
        """Return wire-ready resolution fields plus this display's token/revision."""
        data = (
            self.resolution.to_dict()
            if self.resolution
            else Resolution("", "incomplete").to_dict()
        )
        data.update(token=self.token, revision=self.revision)
        return data

    def commit(self, original, *, token, revision, candidate_index=None):
        """Return the selected full line; None explicitly keeps the original.

        The token is consumed on success. Stale displays, modified arguments and
        invalid indices raise ValueError. feedback_id survives for outcome logging.
        """
        if (
            not self.token
            or self._committed
            or token != self.token
            or revision != self.revision
            or self.resolution is None
            or original != self.resolution.original
            or self._generation != self.resolver.generation
        ):
            raise ValueError("Correction is stale; display the current input again.")
        candidates = self.resolution.candidates
        if candidate_index is not None and (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(candidates)
        ):
            raise ValueError("Invalid correction selection")
        selected = (
            original if candidate_index is None else candidates[candidate_index].text
        )
        if self.resolver.feedback:
            self.resolver.feedback.record_selection(self.feedback_id, candidate_index)
        self._committed = True
        self.token = None
        return selected

    def outcome(self, outcome):
        """Attach a result to the current feedback event when logging is enabled."""
        if self.resolver.feedback:
            self.resolver.feedback.record_outcome(self.feedback_id, outcome)

    def invalidate(self):
        """Expire selection immediately; preserve the previous manual-edit slot."""
        self.token = None
        self.revision += 1


def create_default_resolver(memory=None, enabled=True, shell=None):
    """Build a resolver for the shell used by gated command execution.

    Memory is optional; omitting it gives deterministic ranking without learning.
    PowerShell discovery and a cold Git metadata read happen during construction.
    An invalid/unavailable shell or broken PowerShell catalog is a setup error,
    rather than permission to guess without its aliases and functions.
    """
    if shell is None:
        try:
            from .shell_environment import active_shell_name

            shell = active_shell_name()
        except ImportError:
            shell = "cmd" if os.name == "nt" else "sh"
    if shell not in {"cmd", "sh", "powershell", "pwsh"}:
        raise ValueError("Unsupported execution shell")
    from .shell_environment import shell_executable

    shell_executable(shell)  # a configured but unavailable shell is a setup error
    providers = [IntuitionCommandProvider(), ShellBuiltinProvider(shell)]
    if shell in {"powershell", "pwsh"}:
        from .shell_environment import discover_powershell_commands

        # A broken configured catalog must not silently erase valid aliases and
        # cause the resolver to suggest a different PATH command instead.
        providers.append(
            EnvironmentCatalogProvider(
                discover_powershell_commands(shell), shell=shell, supported_shell=shell
            )
        )
    providers.append(PathExecutableProvider(shell=shell))
    feedback = CorrectionFeedbackStore(memory, enabled) if memory is not None else None
    git_provider = InstalledGitSubcommandProvider()
    git_provider.commands()  # cold metadata read before interactive input
    return CommandResolver(
        providers, shell=shell, feedback=feedback, git_provider=git_provider
    )


_legacy_resolver = CommandResolver([IntuitionCommandProvider()])


def legacy_fuzzy_slash(base):
    """Compatibility only; interfaces must display and bind before submitting."""
    resolution = _legacy_resolver.resolve(base)
    return resolution.candidates[0].text if resolution.candidates else base


def learning_text(text):
    """Argument-free representation for correction-related episode/context logs.

    Call only for command-shaped input. Natural-language retention is an existing
    logging policy of the caller. Git's immediate known subcommand is included;
    everything else, including explicit executable paths, is excluded.
    """
    parts = str(text).strip().split()
    if not parts:
        return ""
    head = _safe_token(parts[0])
    if head == "/exec":
        return "/exec " + _safe_token(parts[1]) if len(parts) > 1 else "/exec"
    if (
        head.casefold() in {"git", "git.exe"}
        and len(parts) > 1
        and parts[1] in GitSubcommandProvider.DEFAULT
    ):
        return head + " " + parts[1]
    return head
