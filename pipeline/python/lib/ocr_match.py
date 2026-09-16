"""
Finding an OCR'd Gurmukhi line in the corpus.

The books quote Gurbani, and every line of Gurbani already sits in
data/corpus.sqlite with its line_id, shabad and ang. So a bold line an engine
read as "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ" is not text to be corrected; it is a line to be
FOUND, after which the corpus text replaces the reading and the line_id rides
along as a citation. The reading only has to be good enough to be recognised.

The same discipline as lib/citations.py, in three layers, and ambiguous is
never guessed:

  L1  the window. The running header names the ang range the page covers
      (lib/ocr_zones.header_hints), give or take ANG_SLACK for a shabad that
      runs over the page break. Inside it there are forty-odd lines, not 60,403.
  L2  similarity inside the window, over grapheme clusters after match_key()
      folds (dandas, counters, nasal marks, nukta). A line that runs on into
      its citation "(ਸਲੋਕ ਮ:੯)" is matched on the best-aligned span, and only
      that span is later replaced.
  L3  the whole corpus, for a quote with no header to narrow it: candidates by
      shared grapheme trigrams, rescored by L2, accepted only at the stricter
      GLOBAL_ACCEPT and never over a close runner-up.

SGGS repeats lines verbatim in different shabads. Two candidates with the same
text are not an ambiguity -- either is the right TEXT -- so the runner-up that
must be beaten by MARGIN is the best candidate with DIFFERENT text, and among
equal texts the one nearest the window is kept.

Grapheme lists are encoded to private-use characters so rapidfuzz's string
alignment routines (partial_ratio_alignment gives the span) work on clusters
rather than code points; a dropped matra is one edit, not two.
"""
from __future__ import annotations
import sqlite3
from collections import Counter, defaultdict

from lib.ocr_text import match_key

ACCEPT_SIM = 0.80          # similarity to accept inside the ang window
GLOBAL_ACCEPT = 0.85       # ... and with no window at all (the margin rule still applies)
MARGIN = 0.05              # by which the best must beat the best different text on the same span
ANG_SLACK = 1
MIN_GRAPHEMES = 8          # shorter keys cannot be told apart
PARTIAL_EXTRA = 2          # a query this many graphemes longer than a line is also scored as containing it
TRIGRAM_TOP = 40


import re as _re
_TITLE_LINE = _re.compile(r"^\s*[੦-੯0-9]+\s*[:.]\s*\S")


class _Codebook:
    def __init__(self):
        self.map: dict[str, str] = {}

    def encode(self, key: list[str]) -> str:
        out = []
        for g in key:
            c = self.map.get(g)
            if c is None:
                c = chr(0xE000 + len(self.map)) if len(self.map) < 6400 else chr(0xF0000 + len(self.map))
                self.map[g] = c
            out.append(c)
        return "".join(out)


class CorpusIndex:
    """Every scripture line by ang, keyed for matching."""

    def __init__(self, rows):
        """rows: iterable of (line_id, shabad_id, ang, text[, source])."""
        self.code = _Codebook()
        self.lines: list[dict] = []
        self.by_ang: dict[int, list[int]] = defaultdict(list)
        self._tri: dict[str, set[int]] | None = None
        for row in rows:
            line_id, shabad_id, ang, text = row[0], row[1], row[2], row[3]
            source = row[4] if len(row) > 4 else "G"
            # BaniDB's Vaaran source carries pauri titles ("੯ : ਗੁਰੂ ਅਮਰਦਾਸ"); a
            # commentary sentence that names the Guru would "quote" them
            if _TITLE_LINE.match(text or ""):
                continue
            key = match_key(text or "")
            if len(key) < 3:
                continue
            idx = len(self.lines)
            self.lines.append({"line_id": line_id, "shabad_id": shabad_id, "ang": ang, "text": text,
                               "source": source, "key": self.code.encode(key)})
            self.by_ang[(source, int(ang))].append(idx)

    @classmethod
    def from_sqlite(cls, con: sqlite3.Connection) -> "CorpusIndex":
        rows = list(con.execute("SELECT line_id, shabad_id, ang, gurmukhi_uni FROM lines ORDER BY line_id"))
        rows = [(*r, "G") for r in rows]
        try:
            rows += list(con.execute("SELECT line_id, shabad_id, ang, text, source FROM granth_lines"))
        except sqlite3.OperationalError:
            pass                                          # the extra sources arrive in M1
        return cls(rows)

    @classmethod
    def from_rows(cls, rows) -> "CorpusIndex":
        return cls(rows)

    def __len__(self) -> int:
        return len(self.lines)

    def window(self, ang_from: int, ang_to: int, source: str = "G", slack: int = ANG_SLACK) -> list[int]:
        out: list[int] = []
        for ang in range(int(ang_from) - slack, int(ang_to) + slack + 1):
            out.extend(self.by_ang.get((source, ang), []))
        return out

    def _trigrams(self) -> dict[str, set[int]]:
        if self._tri is None:
            tri: dict[str, set[int]] = defaultdict(set)
            for i, ln in enumerate(self.lines):
                k = ln["key"]
                for j in range(len(k) - 2):
                    tri[k[j:j + 3]].add(i)
            self._tri = tri
        return self._tri

    def global_candidates(self, q: str, top: int = TRIGRAM_TOP) -> list[int]:
        tri = self._trigrams()
        hits: Counter = Counter()
        for j in range(len(q) - 2):
            for i in tri.get(q[j:j + 3], ()):
                hits[i] += 1
        return [i for i, _ in hits.most_common(top)]


def similarity(q: str, c: str):
    """(score, span) -- span is (start, end) in q when the line is contained in a longer q."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Indel
    full = Indel.normalized_similarity(q, c)
    if len(q) >= len(c) + PARTIAL_EXTRA and len(c) >= SHORT_LINE:
        al = fuzz.partial_ratio_alignment(c, q)
        if al is not None and al.score / 100.0 > full:
            return al.score / 100.0, (al.dest_start, al.dest_end)
    return full, None


# The Jaap Sahib's lines are five or six graphemes ("ਨਮਸਤੰ ਅਗੰਜੇ ॥") and two of
# them share one printed line. A line that short is allowed as a CONTAINED
# match, but only when read almost exactly: chance agreement on six graphemes
# is not rare enough for anything less.
SHORT_LINE = 5
SHORT_ACCEPT = 0.95


def _span_of(span, n: int) -> tuple[int, int]:
    return span if span is not None else (0, n)


def _overlap(a: tuple, b: tuple) -> bool:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return hi - lo > 0.5 * min(a[1] - a[0], b[1] - b[0])


def match_all(key: list[str], index: CorpusIndex, hints: dict | None = None, source: str = "G",
              accept: float = ACCEPT_SIM, margin: float = MARGIN, allow_global: bool = True) -> list[dict]:
    """
    Every corpus line found in this OCR key, in reading order, on
    non-overlapping spans. An OCR line that holds two verse lines ("ਨਾਮ ਕੇ ਧਾਰੇ
    ਸਗਲੇ ਜੰਤ ॥ ਨਾਮ ਕੇ ਧਾਰੇ ਖੰਡ ਬ੍ਰਹਮੰਡ ॥") yields both; a runner-up counts as an
    ambiguity only when it competes for the SAME span with different text.

    @param key   match_key() of the OCR line
    @param hints {"ang_from", "ang_to"} from the header, or None
    @returns [{"line_id", "shabad_id", "ang", "source", "text", "score", "method", "span", "runner_up"}]
    """
    if len(key) < MIN_GRAPHEMES:
        return []
    q = index.code.encode(key)
    n = len(q)
    attempts: list[tuple[str, list[int] | None, float]] = []
    if hints and hints.get("ang_from"):
        cands = index.window(hints["ang_from"], hints.get("ang_to") or hints["ang_from"], source)
        if cands:
            attempts.append(("ang-window", cands, accept))
    # The window is a prior, not a fence: a commentary quotes from the whole
    # Granth, so a line the window does not hold is looked for everywhere, at
    # the stricter global threshold.
    if allow_global:
        attempts.append(("global", None, max(accept, GLOBAL_ACCEPT)))
    for method, cands, threshold in attempts:
        if cands is None:
            cands = index.global_candidates(q)
        if not cands:
            continue
        scored = []
        for i in cands:
            s, span = similarity(q, index.lines[i]["key"])
            if s >= threshold * 0.8:
                scored.append((s, i, _span_of(span, n), span))
        scored.sort(key=lambda t: -t[0])
        taken: list[dict] = []
        for s, i, sp, raw_span in scored:
            if s < threshold:
                break
            if len(index.lines[i]["key"]) < MIN_GRAPHEMES and s < SHORT_ACCEPT:
                continue
            if any(_overlap(sp, t["_span"]) for t in taken):
                continue
            text = index.lines[i]["text"]
            # A competitor must claim a comparable stretch of the reading: the
            # half-line "ਆਦਿ ਪੁਰਖੁ ਨਿਰੰਜਨ ਦੇਉ ॥" of ang 1129 also scores 1.0 inside a
            # two-line quotation from ang 943, but it explains half of it and
            # does not make the whole ambiguous.
            extent = sp[1] - sp[0]
            runner = next((s2 for s2, i2, sp2, _ in scored
                           if i2 != i and index.lines[i2]["text"] != text and _overlap(sp, sp2)
                           and (sp2[1] - sp2[0]) >= 0.8 * extent), 0.0)
            if s - runner < margin:
                continue
            ln = index.lines[i]
            taken.append({"line_id": ln["line_id"], "shabad_id": ln["shabad_id"], "ang": ln["ang"],
                          "source": ln["source"], "text": text, "score": round(s, 3), "method": method,
                          "span": raw_span, "runner_up": round(runner, 3), "_span": sp})
        if taken:
            taken.sort(key=lambda t: t["_span"][0])
            for t in taken:
                del t["_span"]
            return taken
    return []


def match_line(key: list[str], index: CorpusIndex, hints: dict | None = None, source: str = "G",
               accept: float = ACCEPT_SIM, margin: float = MARGIN, allow_global: bool = True) -> dict | None:
    """The best single corpus line in this OCR key, or None (see match_all)."""
    found = match_all(key, index, hints, source, accept, margin, allow_global)
    if not found:
        return None
    return max(found, key=lambda m: m["score"])


def match_text(text: str, index: CorpusIndex, hints: dict | None = None, **kw) -> dict | None:
    return match_line(match_key(text), index, hints, **kw)


def match_text_all(text: str, index: CorpusIndex, hints: dict | None = None, **kw) -> list[dict]:
    return match_all(match_key(text), index, hints, **kw)


def align_stream(keys: list[list[str]], index: CorpusIndex, hints: dict | None = None, source: str = "G",
                 accept: float = ACCEPT_SIM, margin: float = MARGIN) -> list[dict]:
    """
    Corpus lines found across a RUN of OCR lines read as one stream.

    In the Santhya's two-column pages the ang's Gurbani is set bold in a
    narrow left column, so one corpus line wraps over two or three OCR lines
    and no single OCR line holds enough of it to match. The keys of the run
    are concatenated and matched as one text; each match comes back with the
    OCR lines it covers and the grapheme range it covers in each.

    @returns [{... match fields ..., "lines": [(line index, start, end), ...]}]
             where start/end are offsets into that line's key
    """
    offsets, stream = [], []
    for k in keys:
        offsets.append(len(stream))
        stream.extend(k)
    if len(stream) < MIN_GRAPHEMES:
        return []
    found = match_all(stream, index, hints, source, accept, margin)
    out = []
    for m in found:
        s, e = m["span"] if m["span"] else (0, len(stream))
        covered = []
        for i, off in enumerate(offsets):
            end = offsets[i + 1] if i + 1 < len(offsets) else len(stream)
            lo, hi = max(s, off), min(e, end)
            if hi > lo:
                covered.append((i, lo - off, hi - off))
        if covered:
            out.append({**m, "lines": covered})
    return out
