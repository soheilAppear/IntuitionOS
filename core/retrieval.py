"""Retrieval: getting saved notes into the prompt, and doing it without being asked.

The README said that after `/save`, the model uses your notes as context. It did
not. `Brain.step` built `msgs = [system, user]` and nothing else, and
`Memory.search` was `LIKE %term%` — which cannot use an index, does no ranking,
and matches the middle of words.

Two things are fixed here, and the second is the interesting one.

The mechanical fix is FTS5: a real full-text index with BM25 ranking, migrated
from the existing `mem` table without touching it, so a user's notes survive.

The design fix is that retrieval becomes cue-driven. A note surfaces because the
situation resembles the one it was written in, not because the user typed
`/recall`. That is what episodic memory does — you do not query it, it presents
itself — and it is the difference between a searchable file and something that
feels like it remembers.

Everything is bounded by a token budget, because a database with ten thousand
notes must not produce a prompt that will not fit.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

# A rough characters-per-token ratio. Deliberately conservative: overestimating
# the budget produces a prompt the model silently truncates from the front, which
# is the worst possible failure because it eats the system prompt first.
CHARS_PER_TOKEN = 3.5

DEFAULT_BUDGET_TOKENS = 700
# Notes decay in relevance more slowly than commands do; a month is about right
# for "this is still what I am working on".
RECENCY_HALF_LIFE_S = 30 * 24 * 3600.0

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(
  text,
  content='mem',
  content_rowid='id',
  tokenize='unicode61'
);
"""

# Keep the index in step with the table it shadows. Without these an edit or a
# delete would leave the index asserting something that is no longer true.
FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS mem_fts_insert AFTER INSERT ON mem BEGIN
  INSERT INTO mem_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS mem_fts_delete AFTER DELETE ON mem BEGIN
  INSERT INTO mem_fts(mem_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS mem_fts_update AFTER UPDATE ON mem BEGIN
  INSERT INTO mem_fts(mem_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO mem_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

# Words that would match half the database and rank nothing.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "by", "from", "as", "it", "its",
    "i", "me", "my", "you", "your", "we", "our", "do", "does", "did", "how",
    "what", "when", "where", "which", "who", "why", "can", "could", "should",
    "would", "will", "shall", "have", "has", "had", "not", "no", "yes", "so",
}

_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")


@dataclass
class Retrieved:
    id: int
    ts: float
    role: str
    text: str
    tags: str
    score: float
    reason: str = ""

    def age_days(self) -> float:
        return max(0.0, (time.time() - self.ts) / 86400.0)


class Retriever:
    """FTS5-backed retrieval over the notes table, with recency weighting."""

    def __init__(self, memory, budget_tokens: int = DEFAULT_BUDGET_TOKENS):
        self.mem = memory
        self.budget_tokens = budget_tokens
        self.available = self._migrate()

    # ── Migration ────────────────────────────────────────────────────────

    def _migrate(self) -> bool:
        """Build the index beside the existing table, then backfill it.

        FTS5 is compiled into most SQLite builds but not all, so its absence
        degrades to LIKE rather than taking the process down. `content='mem'`
        makes it an external-content index: the note text is not duplicated, and
        `mem` remains the single source of truth.
        """
        # Whether the index is new has to be decided *before* creating it, and it
        # cannot be decided by counting rows afterwards: an external-content FTS5
        # table reads through to `mem`, so SELECT COUNT(*) FROM mem_fts always
        # equals SELECT COUNT(*) FROM mem whether or not anything is indexed.
        already_existed = self.mem.has_table("mem_fts")
        try:
            self.mem.executescript(FTS_SCHEMA)
            self.mem.executescript(FTS_TRIGGERS)
        except Exception:
            return False
        if not already_existed:
            try:
                self.reindex()
            except Exception:
                return False
        return True

    def reindex(self) -> None:
        """Build the index from `mem`. Run once on creation, since the triggers
        keep it current from then on."""
        self.mem.execute("INSERT INTO mem_fts(mem_fts) VALUES ('rebuild')")

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10, roles: Optional[tuple] = None) -> list:
        """Ranked full-text search, falling back to LIKE where FTS is absent."""
        terms = self._terms(query)
        if not terms:
            return []
        if not self.available:
            return self._like_search(query, limit, roles)
        try:
            return self._fts_search(terms, limit, roles)
        except Exception:
            return self._like_search(query, limit, roles)

    def _fts_search(self, terms: list, limit: int, roles: Optional[tuple]) -> list:
        # OR rather than AND: a note matching two of five words is usually the
        # one wanted, and requiring all of them makes retrieval silent far too
        # often to be useful as an ambient cue.
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = (
            "SELECT m.id, m.ts, m.role, m.text, m.tags, bm25(mem_fts) AS rank"
            " FROM mem_fts JOIN mem m ON m.id = mem_fts.rowid"
            " WHERE mem_fts MATCH ?"
        )
        params: list = [match]
        if roles:
            sql += " AND m.role IN (%s)" % ",".join("?" * len(roles))
            params += list(roles)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit * 3)  # over-fetch, then re-rank by recency below

        rows = self.mem.query(sql, tuple(params))
        # bm25 returns lower-is-better, so it is negated into a positive score.
        return self._rerank([
            Retrieved(id=r[0], ts=r[1], role=r[2], text=r[3], tags=r[4], score=-float(r[5]))
            for r in rows
        ], limit)

    def _like_search(self, query: str, limit: int, roles: Optional[tuple]) -> list:
        terms = self._terms(query)
        found: dict = {}
        for term in terms[:5]:
            sql = "SELECT id, ts, role, text, tags FROM mem WHERE text LIKE ?"
            params: list = [f"%{term}%"]
            if roles:
                sql += " AND role IN (%s)" % ",".join("?" * len(roles))
                params += list(roles)
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(limit * 3)
            for r in self.mem.query(sql, tuple(params)):
                entry = found.get(r[0])
                if entry:
                    entry.score += 1.0
                else:
                    found[r[0]] = Retrieved(id=r[0], ts=r[1], role=r[2], text=r[3],
                                            tags=r[4], score=1.0)
        return self._rerank(list(found.values()), limit)

    def _rerank(self, items: list, limit: int) -> list:
        """Blend textual relevance with recency.

        A perfect lexical match on a note from a year ago is usually less useful
        than a decent match on one from this week, so recency multiplies rather
        than merely tie-breaks.
        """
        now = time.time()
        for item in items:
            age = max(0.0, now - (item.ts or now))
            recency = 0.5 ** (age / RECENCY_HALF_LIFE_S)
            item.score = item.score * (0.5 + 0.5 * recency)
            item.reason = f"matched, {item.age_days():.0f}d old"
        items.sort(key=lambda i: -i.score)
        return items[:limit]

    @staticmethod
    def _terms(query: str) -> list:
        words = [w.lower() for w in _WORD.findall(query or "")]
        terms = [w for w in words if w not in _STOPWORDS and len(w) > 1]
        # A query that is nothing but stopwords should still search for something.
        return (terms or words)[:8]

    # ── Cue-driven retrieval ─────────────────────────────────────────────

    def retrieve(self, query: str, ctx=None, k: int = 4,
                 budget_tokens: Optional[int] = None) -> list:
        """Notes worth putting in front of the model right now.

        The cue is the user's text *plus* the situation, which is what makes this
        different from `/recall`. A note mentioning the branch you are on, or the
        file you just edited, surfaces because you are there — not because you
        thought to ask for it.
        """
        cue = " ".join(filter(None, [query, _context_cue(ctx)]))
        notes = self.search(cue, limit=k * 2, roles=("note",))
        return self._fit_budget(notes, budget_tokens or self.budget_tokens, k)

    def _fit_budget(self, notes: list, budget_tokens: int, k: int) -> list:
        """Take the best notes that fit. A 10,000-note database must not be able
        to produce an oversized prompt."""
        budget_chars = int(budget_tokens * CHARS_PER_TOKEN)
        kept, used = [], 0
        for note in notes:
            if len(kept) >= k:
                break
            cost = len(note.text) + 8  # bullet, newline, a little framing
            if used + cost > budget_chars:
                continue  # a long note is skipped, not truncated into nonsense
            kept.append(note)
            used += cost
        return kept


def _context_cue(ctx) -> str:
    """Turn the situation into search terms.

    Deliberately narrow. Throwing the whole Context at the index would match
    everything; the branch, the basename of the directory and the last command
    are the parts that identify *what you are doing*.
    """
    if ctx is None:
        return ""
    parts = []
    branch = getattr(ctx, "git_branch", None)
    if branch:
        parts.append(str(branch))
    cwd = getattr(ctx, "cwd", "") or ""
    if cwd:
        parts.append(re.split(r"[\\/]", cwd.rstrip("\\/"))[-1])
    recent = getattr(ctx, "recent_commands", None) or []
    if recent:
        last = recent[-1]
        text = last.get("text") if isinstance(last, dict) else str(last)
        if text:
            parts.append(text)
    for path in (getattr(ctx, "recent_files", None) or [])[:2]:
        parts.append(re.split(r"[\\/]", str(path).rstrip("\\/"))[-1])
    return " ".join(parts)


def render_notes(notes: list) -> str:
    """The block injected into the prompt."""
    if not notes:
        return ""
    lines = ["RELEVANT NOTES YOU SAVED EARLIER"]
    for n in notes:
        lines.append(f"- ({n.age_days():.0f}d ago) {n.text}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1
