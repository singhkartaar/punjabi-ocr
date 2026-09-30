"""
The notation record: what one parsed notation looks like, and the rules it obeys.

A notation is script-neutral. What is stored is the music -- swaras with
their octave, komal and tivra marks, held beats, rests, groups of notes in
one beat -- and the sung syllables in Gurmukhi with their held marks; every
row a reader sees (Gurmukhi swaras, Latin swaras, roman syllables, the
sam/khali markers) is derived from that by lib/notation_render.py and its
JavaScript twin. So the same record serves the review page, the web app and
the English rendering without a second copy of anything.

The shape is documented in lib/notation.schema.json; validate() here is
the hand-written check both sides run (the JavaScript one carries the same
error codes), because the rules that matter cross fields: a beat is one of
notes / unreadable / held / rest, the fractions of a beat add up, komal only
on R G D N, tivra only on M, matras consecutive from where the row starts,
bol spans increasing. A record that fails is not written.

Defaults are omitted on write and filled on read (fill_defaults), so the
files stay small: a shuddh madhya note is {"s": "P"}.
"""
from __future__ import annotations
import hashlib
import json
import os
import re

from lib.notation_vocab import KOMAL_ALLOWED, RAAGS, SWARA_ORDER, TAALS, TIVRA_ALLOWED, VERSION as VOCAB_VERSION

SCHEMA_VERSION = 1
PARSER_VERSION = "0.1.0"

KINDS = ("notation", "partial", "reet-ref", "non-gurbani")
SCRIPTS = ("gurmukhi", "latin", "devanagari")
SECTION_KINDS = ("sthai", "antara", "sanchari", "abhog", "alaap", "taan", "tihai", "other")
LINE_KINDS = ("avartan", "free")
DIVS = (1, 2, 3, 4, 6, 8)
SOURCES = ("G", "D", "B", "K", "N")
RESOLVE_METHODS = ("stream+ref+bol", "stream+ref", "stream+bol", "stream", "ref-window", "ref+bol", "bol", "book-ref", "manual", "none")
FLAGS = frozenset([
    "unresolved-shabad", "weak-shabad", "ref-conflict", "no-shabad-text", "no-section-label",
    "taal-mismatch", "taal-unknown", "raag-unknown", "style-contradiction", "continues-next-page",
    "continued-from-prev", "empty-beat", "unread-cell", "tick-on-non-ma", "diagonal-watermark",
    "partial-grid", "unmatched-text", "taal-changes", "raag-differs", "shabad-by-book-ref",
    "shabad-inherited", "span-capped", "shabad-by-bol",
])
DEFAULT_STYLE = {"swar_row": "above", "shabad_position": "before", "matra_row": False,
                 "marker_row": "below", "table": "bars", "labels": False, "script": "gurmukhi"}
STYLE_VALUES = {"swar_row": ("above", "below"), "shabad_position": ("before", "after", "either"),
                "marker_row": ("below", "above", "none"), "table": ("ruled", "bars", "none"),
                "script": SCRIPTS}

BOOK_KEY_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
ID_RE = re.compile(r"^([a-z0-9]+(?:-[a-z0-9]+)*):(\d{4}):(\d+)$")

NOTE_DEFAULTS = {"o": 0, "k": False, "t": False, "len": 1, "kh": False}
BEAT_DEFAULTS = {"div": 1}
LINE_DEFAULTS = {"matra_from": 1, "continues": False}


# ---- identifiers -----------------------------------------------------------

def mid_window(n_pages: int, window: int = 5, at: float = 0.45) -> tuple[int, int]:
    """
    The review protocol's first window: `window` consecutive pages starting
    at about 45% of the book (1-based, inclusive). The reviewer's run
    extends it, a page at a time, until two resolved notations are covered.
    """
    n = max(1, int(n_pages))
    start = max(1, min(n, int(round(n * at))))
    end = min(n, start + max(1, window) - 1)
    return start, end


def notation_slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "unknown"


def make_id(book_key: str, page: int, seq: int) -> str:
    """'gurmat-sangeet-sagar-1:0042:1' -- the book, the rendered page, the grid's order on it."""
    if not BOOK_KEY_RE.match(book_key):
        raise ValueError("book key %r is not a slug" % book_key)
    return "%s:%04d:%d" % (book_key, page, seq)


def parse_id(notation_id: str) -> tuple[str, int, int] | None:
    m = ID_RE.match(notation_id or "")
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def image_name(notation_id: str, n: int, thumb: bool = False) -> str:
    """The file a crop is written to: colons are not for file names."""
    return "%s-%d%s.png" % (notation_id.replace(":", "-"), n, ".thumb" if thumb else "")


# ---- style -----------------------------------------------------------------

def merge_style(*layers: dict | None) -> dict:
    """DEFAULT_STYLE under a manifest's top-level style under a work's; unknown values are errors."""
    style = dict(DEFAULT_STYLE)
    for layer in layers:
        for k, v in (layer or {}).items():
            if k not in DEFAULT_STYLE:
                raise ValueError("unknown style key %r" % k)
            if k in STYLE_VALUES and v not in STYLE_VALUES[k]:
                raise ValueError("style %s must be one of %s, not %r" % (k, STYLE_VALUES[k], v))
            style[k] = v
    return style


# ---- defaults and canonical form -------------------------------------------

def fill_defaults(rec: dict) -> dict:
    """The record with every omitted default present. Returns a deep copy."""
    out = json.loads(json.dumps(rec))
    for sec in out.get("sections") or []:
        for line in sec.get("lines") or []:
            for k, v in LINE_DEFAULTS.items():
                line.setdefault(k, v)
            for beat in line.get("beats") or []:
                for k, v in BEAT_DEFAULTS.items():
                    beat.setdefault(k, v)
                for note in beat.get("notes") or []:
                    for k, v in NOTE_DEFAULTS.items():
                        note.setdefault(k, v)
                    if note.get("kan"):
                        for k, v in NOTE_DEFAULTS.items():
                            if k != "len":
                                note["kan"].setdefault(k, v)
                bol = beat.get("bol")
                if isinstance(bol, dict):
                    bol.setdefault("h", 0)
    return out


def strip_defaults(rec: dict) -> dict:
    """The inverse: defaults removed, so the written file is small. Returns a deep copy."""
    out = json.loads(json.dumps(rec))
    for sec in out.get("sections") or []:
        for line in sec.get("lines") or []:
            for k, v in LINE_DEFAULTS.items():
                if line.get(k) == v:
                    line.pop(k, None)
            for beat in line.get("beats") or []:
                if beat.get("div") == 1:
                    beat.pop("div", None)
                for note in beat.get("notes") or []:
                    for k, v in NOTE_DEFAULTS.items():
                        if note.get(k) == v:
                            note.pop(k, None)
                    if note.get("kan"):
                        for k, v in NOTE_DEFAULTS.items():
                            if note["kan"].get(k) == v:
                                note["kan"].pop(k, None)
                bol = beat.get("bol")
                if isinstance(bol, dict) and bol.get("h") == 0:
                    bol.pop("h", None)
    return out


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(rec: dict) -> str:
    """sha1 of what the music says: sections, the shabad, the raag used and the taal."""
    core = {"sections": strip_defaults(rec).get("sections"),
            "shabad_id": (rec.get("shabad") or {}).get("shabad_id"),
            "raag_used": ((rec.get("heading") or {}).get("raag") or {}).get("key"),
            "taal": ((rec.get("heading") or {}).get("taal") or {}).get("key")}
    return hashlib.sha1(canonical_json(core).encode("utf-8")).hexdigest()


# ---- validation ------------------------------------------------------------

def validate(rec: dict) -> list[dict]:
    """
    Every rule the record breaks, as [{"code", "path", "detail"}]; empty when
    it is sound. The codes are shared with the JavaScript validator.
    """
    errs: list[dict] = []

    def err(code, path, detail=""):
        errs.append({"code": code, "path": path, "detail": detail})

    if rec.get("schema_version") != SCHEMA_VERSION:
        err("schema.version", "schema_version", "expected %d" % SCHEMA_VERSION)
    nid = rec.get("notation_id") or ""
    parsed = parse_id(nid)
    if not parsed:
        err("id.format", "notation_id", nid)
    else:
        book, page, seq = parsed
        if book != rec.get("book_key") or page != rec.get("page") or seq != rec.get("seq"):
            err("id.format", "notation_id", "id disagrees with book_key/page/seq")
    if rec.get("kind") not in KINDS:
        err("enum.kind", "kind", str(rec.get("kind")))
    if rec.get("script") not in SCRIPTS:
        err("enum.script", "script", str(rec.get("script")))
    pages = rec.get("pages")
    if not isinstance(pages, list) or not pages or rec.get("page") not in pages:
        err("pages.list", "pages", "must list every page and include `page`")
    for f in rec.get("flags") or []:
        if f not in FLAGS:
            err("flags.unknown", "flags", f)

    heading = rec.get("heading") or {}
    top_taal = heading.get("taal")
    rk = (heading.get("raag") or {}).get("key")
    if rk is not None and rk not in RAAGS:
        err("vocab.key", "heading.raag.key", rk)
    if top_taal is not None and top_taal.get("key") is not None and top_taal["key"] not in TAALS:
        err("vocab.key", "heading.taal.key", top_taal["key"])
    rs = rec.get("raag_shabad")
    if rs is not None and rs not in RAAGS:
        err("vocab.key", "raag_shabad", rs)
    shabad = rec.get("shabad") or {}
    if shabad.get("source") is not None and shabad["source"] not in SOURCES:
        err("enum.source", "shabad.source", str(shabad.get("source")))
    if shabad.get("method") not in RESOLVE_METHODS:
        err("enum.method", "shabad.method", str(shabad.get("method")))
    if shabad.get("shabad_id") is None and shabad.get("method") not in ("none", "manual"):
        err("shabad.unresolved", "shabad", "no shabad_id but method says it was resolved")

    sections = rec.get("sections")
    # a partial record (the grid not yet read, or unreadable) may have no sections
    if rec.get("kind") == "notation" and (not isinstance(sections, list) or not sections):
        err("sections.empty", "sections")
    if rec.get("kind") == "partial" and not sections and "partial-grid" not in (rec.get("flags") or []):
        err("sections.empty", "sections", "a partial record without sections must say partial-grid")
    seen: dict[tuple[str, int], int] = {}
    for si, sec in enumerate(sections or []):
        sp = "sections[%d]" % si
        if sec.get("kind") not in SECTION_KINDS:
            err("enum.section", sp + ".kind", str(sec.get("kind")))
        key = (sec.get("kind"), sec.get("n"))
        if sec.get("n") is not None:
            if key in seen:
                err("section.n", sp + ".n", "duplicate %s %s" % key)
            seen[key] = si
        sec_taal = sec.get("taal") or top_taal
        taal_key = (sec_taal or {}).get("key")
        if sec.get("taal") and sec["taal"].get("key") is not None and sec["taal"]["key"] not in TAALS:
            err("vocab.key", sp + ".taal.key", sec["taal"]["key"])
        matras = TAALS[taal_key]["matras"] if taal_key in TAALS else (sec_taal or {}).get("matras")
        lines = sec.get("lines")
        if not isinstance(lines, list) or not lines:
            err("lines.empty", sp + ".lines")
            continue
        for li, line in enumerate(lines):
            lp = "%s.lines[%d]" % (sp, li)
            kind = line.get("kind")
            if kind not in LINE_KINDS:
                err("enum.line", lp + ".kind", str(kind))
            beats = line.get("beats")
            if not isinstance(beats, list) or not beats:
                err("beats.empty", lp + ".beats")
                continue
            if kind == "avartan" and not sec_taal:
                err("line.taal", lp, "an avartan line needs a taal")
            expect = line.get("matra_from", 1)
            spans: list[tuple[int, int]] = []
            for bi, beat in enumerate(beats):
                bp = "%s.beats[%d]" % (lp, bi)
                _validate_beat(beat, bp, kind, err)
                if kind == "avartan":
                    m = beat.get("m")
                    if m != expect:
                        err("line.matras", bp + ".m", "expected %s, got %s" % (expect, m))
                    expect = (m if isinstance(m, int) else expect) + 1
                elif "m" in beat:
                    err("line.matras", bp + ".m", "a free line has no matras")
                bol = beat.get("bol")
                if isinstance(bol, dict) and bol.get("span") is not None:
                    span = bol["span"]
                    if line.get("line_id") is None:
                        err("bol.span", bp + ".bol.span", "a span needs the line's line_id")
                    elif (not isinstance(span, list) or len(span) != 2 or span[0] >= span[1]
                          or (spans and span[0] < spans[-1][1])):
                        err("bol.span", bp + ".bol.span", "spans must be increasing and non-overlapping")
                    else:
                        spans.append((span[0], span[1]))
            if kind == "avartan" and matras and expect - 1 > matras:
                err("line.matras", lp, "last matra %d exceeds the taal's %d" % (expect - 1, matras))
    return errs


def _validate_beat(beat: dict, bp: str, line_kind: str | None, err) -> None:
    has_notes = "notes" in beat
    modes = [has_notes, bool(beat.get("ext")), bool(beat.get("rest"))]
    if sum(modes) != 1:
        err("beat.exclusive", bp, "exactly one of notes / ext / rest")
    div = beat.get("div", 1)
    if div not in DIVS:
        err("beat.div", bp + ".div", str(div))
    if has_notes:
        notes = beat["notes"]
        if notes is None:
            if not beat.get("raw") or beat.get("c") is None:
                err("beat.unknown_raw", bp, "an unreadable beat keeps its raw text and confidence")
        elif not isinstance(notes, list) or not notes:
            err("beat.exclusive", bp + ".notes", "notes must be a non-empty list or null")
        else:
            total = 0
            for ni, note in enumerate(notes):
                total += _validate_note(note, "%s.notes[%d]" % (bp, ni), err, allow_len=True)
            if total != div:
                err("beat.len_sum", bp, "lengths sum to %d, div is %d" % (total, div))
    bol = beat.get("bol")
    if bol is not None and "bol" in beat:
        if not isinstance(bol, dict) or not isinstance(bol.get("g", ""), str):
            err("bol.shape", bp + ".bol")
        elif not isinstance(bol.get("h", 0), int) or bol.get("h", 0) < 0:
            err("bol.shape", bp + ".bol.h")


def _validate_note(note: dict, path: str, err, allow_len: bool) -> int:
    s = note.get("s")
    if s not in SWARA_ORDER:
        err("note.swar", path + ".s", str(s))
        return note.get("len", 1) if isinstance(note.get("len", 1), int) else 1
    if note.get("k") and s not in KOMAL_ALLOWED:
        err("note.komal", path, "%s has no komal" % s)
    if note.get("t") and s not in TIVRA_ALLOWED:
        err("note.tivra", path, "%s has no tivra" % s)
    o = note.get("o", 0)
    if not isinstance(o, int) or o < -2 or o > 2:
        err("note.octave", path + ".o", str(o))
    ln = note.get("len", 1)
    if not allow_len and "len" in note:
        err("note.kan", path, "a kan has no length")
    if not isinstance(ln, int) or ln < 1:
        err("beat.len_sum", path + ".len", str(ln))
        ln = 1
    kan = note.get("kan")
    if kan is not None:
        if not isinstance(kan, dict):
            err("note.kan", path + ".kan")
        else:
            _validate_note(kan, path + ".kan", err, allow_len=False)
            if kan.get("kan") is not None:
                err("note.kan", path + ".kan", "a kan has no kan")
    return ln


# ---- files -----------------------------------------------------------------

def apply_gold(rec: dict, gold: dict | None) -> dict:
    """
    The reviewer's verdict on a record (30_notation_gt.py --promote): the
    record is marked verified, and a field the reviewer corrected takes the
    correction -- the shabad, the raag used, the taal, the laya. Cell
    corrections are applied to the parsed grid where it has the cell. The
    record is changed in place and returned; no gold, no change.
    """
    if not gold or gold.get("status") == "skip" or not gold.get("verified", True):
        return rec
    def truth(field):
        j = gold.get(field) or {}
        if "truth" in j:
            return j["truth"]
        return j.get("value") if j.get("ok") else (j.get("correct") or None)
    sh = rec.setdefault("shabad", {})
    t = truth("shabad")
    if (gold.get("shabad") or {}).get("ok") is False:
        sh["shabad_id"] = int(t) if t not in (None, "") else None
        sh["method"] = "manual"
        sh["confidence"] = 1.0 if sh["shabad_id"] is not None else 0.0
    sh["verified"] = (gold.get("shabad") or {}).get("ok") in (True, False)
    heading = rec.get("heading")
    if heading is not None:
        if (gold.get("raag_used") or {}).get("ok") is False:
            heading.setdefault("raag", {})["key"] = truth("raag_used")
            heading["raag"]["confidence"] = 1.0
        if (gold.get("taal") or {}).get("ok") is False:
            heading.setdefault("taal", {})["key"] = truth("taal")
            heading["taal"]["confidence"] = 1.0
        if (gold.get("laya") or {}).get("ok") is False and heading.get("taal") is not None:
            heading["taal"]["laya"] = truth("laya")
    for c in gold.get("cells") or []:
        try:
            beat = rec["sections"][int(c["s"])]["lines"][int(c["l"])]["beats"][int(c["b"])]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        f, v = c.get("field"), c.get("value")
        if f == "bol":
            beat["bol"] = {"g": v or "", "h": 0} if isinstance(v, str) else v
        elif f in ("m", "div"):
            beat[f] = int(v)
        elif f in ("notes", "ext", "rest"):
            from lib.notation_render import parse_cell
            parsed = parse_cell(str(v)) if isinstance(v, str) else v
            if isinstance(parsed, dict):
                for k in ("notes", "ext", "rest"):
                    beat.pop(k, None)
                beat.update(parsed)
    rec["verified"] = True
    rec["source"]["gold"] = {"by": gold.get("by"), "at": gold.get("at"), "status": gold.get("status", "ok")}
    rec["source"]["content_hash"] = content_hash(rec)
    return rec


def write_jsonl(path: str, meta: dict, records: list[dict]) -> None:
    """`_meta` first, one record a line, written whole then renamed into place."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for rec in records:
            fh.write(json.dumps(strip_defaults(rec), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows or "_meta" not in rows[0]:
        raise ValueError("%s: no _meta line" % path)
    return rows[0]["_meta"], rows[1:]


def meta_for(book: dict, **extra) -> dict:
    """The `_meta` line of a book's notations.jsonl."""
    return {"schema": SCHEMA_VERSION, "parser_version": PARSER_VERSION, "vocab_version": VOCAB_VERSION,
            **book, **extra}
