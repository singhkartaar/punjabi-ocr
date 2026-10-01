"""
The index (ਤਤਕਰਾ, ਸੂਚੀ-ਪੱਤਰ) a keertan book opens with, read for what it says
about the notations.

Tara Singh's ratnavalis list every shabad's first line with its printed
page number, under numbered raag headings ("9. ਰਾਗ ਸੋਰਠਿ ਪਰਿਚਯ ਅਤੇ ਸੁਰ
ਵਿਸਤਾਰ 56", then "ਰੇ ਮਨ ਰਾਮ ਸਿਉ ਕਰਿ ਪ੍ਰੀਤਿ 58"); Dyal Singh's Sangeet Sagar
numbers its notations and lists them the same way. Tesseract reads such a
page as lines that end in a number, or as a text line with the number
standing alone on the same row at the right margin.

What comes out: [{"number", "text", "page_printed", "shabad_id", "score",
"kind": "shabad" | "section"}], one entry per line of the index, the
shabad named by the corpus matcher where the first line is one it knows.
The parser uses it two ways, both additive: a numbered notation printed
without its verse takes the shabad of index entry N (method "index"),
and a resolved notation whose number's entry names another shabad is
flagged (index-conflict), never overridden. The expected count of
notations in a book is the count of shabad entries.
"""
from __future__ import annotations
import re

from lib.notation_vocab import gurmukhi_digits

_TRAIL_NUM = re.compile(r"[\s.()]*([0-9੦-੯]{1,4})[\s.)]*$")
_LEAD_NUM = re.compile(r"^\s*([0-9੦-੯]{1,3})\s*[.)]\s*")
_NUM_ONLY = re.compile(r"^[\s.()]*[0-9੦-੯]{1,4}[\s.()]*$")
_SECTION_WORDS = re.compile(r"ਪਰਿਚਯ|ਪਰਿਚੈ|ਵਿਸਤਾਰ|ਭੂਮਿਕਾ|ਮੁੱਖ|ਤਤਕਰਾ|ਸੂਚੀ|ਜਾਣ-ਪਛਾਣ|ਚਿੰਨ੍ਹ|ਤਾਲ ਅਤੇ|ਪ੍ਰਸਤਾਵਨਾ|ਸ਼ਬਦ ਸੂਚੀ|ਅਨੁਕ੍ਰਮਣਿਕਾ")
_TITLE_WORDS = re.compile(r"ਤਤਕਰਾ|ਸੂਚੀ|ਅਨੁਕ੍ਰਮਣਿਕਾ|ਵਿਸ਼ਾ-ਸੂਚੀ")
_GURMUKHI = re.compile(r"[ਅ-ਹਖ਼-ਫ਼]")

MIN_ENTRIES = 8             # an index page lists this many lines with a page number at least
MIN_SHARE = 0.4             # ... and they are this share of its body lines
FRONT_PAGES = 25            # the index is looked for in the first pages of a book ...
BACK_PAGES = 20             # ... and in the last ones (some books put it there)


def _rows(lines: list[dict], tol: int = 28) -> list[list[dict]]:
    """Body lines grouped into printed rows by their vertical centre: a text line and the number at its right margin."""
    body = sorted((l for l in lines if l.get("bbox") and l.get("zone") in (None, "body")),
                  key=lambda l: ((l["bbox"][1] + l["bbox"][3]) / 2, l["bbox"][0]))
    rows: list[list[dict]] = []
    for l in body:
        cy = (l["bbox"][1] + l["bbox"][3]) / 2
        if rows and abs(cy - (rows[-1][0]["bbox"][1] + rows[-1][0]["bbox"][3]) / 2) <= tol:
            rows[-1].append(l)
        else:
            rows.append([l])
    return [sorted(r, key=lambda l: l["bbox"][0]) for r in rows]


def row_entry(row: list[dict]) -> dict | None:
    """
    One printed row of an index as {"number", "text", "page_printed", "kind"},
    or None when the row carries no page number. The text is what stands
    before the page number; a leading "9." is the section number.
    """
    texts = [(l.get("text") or "").strip() for l in row]
    texts = [t for t in texts if t]
    if not texts:
        return None
    page = None
    # the number standing alone at the right
    if len(texts) >= 2 and _NUM_ONLY.match(texts[-1]):
        page = int(gurmukhi_digits(re.sub(r"[^0-9੦-੯]", "", texts[-1])))
        texts = texts[:-1]
    text = " ".join(texts)
    if page is None:
        m = _TRAIL_NUM.search(text)
        if not m or not _GURMUKHI.search(text[:m.start()]):
            return None
        page = int(gurmukhi_digits(m.group(1)))
        text = text[:m.start()].strip()
    number = None
    m = _LEAD_NUM.match(text)
    if m:
        number = int(gurmukhi_digits(m.group(1)))
        text = text[m.end():].strip()
    text = text.strip(" .:,-—'\"")
    if not _GURMUKHI.search(text) or len(text) < 4:
        return None
    kind = "section" if _SECTION_WORDS.search(text) or (number is not None and re.match(r"^ਰਾਗ", text)) else "shabad"
    return {"number": number, "text": text, "page_printed": page, "kind": kind}


def page_entries(lines: list[dict]) -> list[dict]:
    """The index entries on one page's merged lines, in reading order; [] when the page is no index."""
    rows = _rows(lines)
    entries = [e for e in (row_entry(r) for r in rows) if e]
    body = sum(1 for r in rows if any((l.get("text") or "").strip() for l in r))
    if len(entries) < MIN_ENTRIES or len(entries) < MIN_SHARE * max(1, body):
        return []
    return entries


def find_index(pages: dict[int, list[dict]]) -> dict:
    """
    {"pages": [rendered page numbers], "entries": [...]} over the merged
    lines of a book's first pages ({page: lines}). A run of index pages is
    taken whole; a page that merely has many numbered lines (a table of
    taals) counts only when a neighbour or itself names the index, or when
    at least two consecutive pages qualify.
    """
    found: list[tuple[int, list[dict], bool]] = []
    for p in sorted(pages):                       # the caller chooses the pages: the front of the book, the back
        lines = pages[p]
        entries = page_entries(lines)
        titled = any(_TITLE_WORDS.search(l.get("text") or "") for l in lines)
        if entries:
            found.append((p, entries, titled))
    if not found:
        return {"pages": [], "entries": []}
    keep: list[tuple[int, list[dict]]] = []
    for k, (p, entries, titled) in enumerate(found):
        prev_adjacent = k > 0 and found[k - 1][0] == p - 1
        next_adjacent = k + 1 < len(found) and found[k + 1][0] == p + 1
        if titled or prev_adjacent or next_adjacent or len(entries) >= 2 * MIN_ENTRIES:
            keep.append((p, entries))
    out: list[dict] = []
    titled_pages = {p for p, _, t in found if t}
    for p, entries in keep:
        for e in entries:
            out.append({**e, "index_page": p})
    # a section's number carries to the shabads under it until the next section; a section that
    # names a raag is where that raag begins in the book (a raag-only index still says that)
    from lib.notation_text import parse_heading
    section = None
    for e in out:
        if e["kind"] == "section":
            section = e["number"] if e["number"] is not None else section
            raag = (parse_heading(e["text"]) or {}).get("raag") or {}
            if raag.get("key"):
                e["raag_key"] = raag["key"]
        else:
            e["section"] = section
    # Dyal Singh numbers the notations themselves: a shabad entry with its own number keeps it
    return {"pages": [p for p, _ in keep], "entries": out, "titled": bool(titled_pages & {p for p, _ in keep})}


def credible(index: dict, matched: int, min_share: float = 0.25, min_matched: int = 3) -> bool:
    """
    An index is believed when the corpus knows a fair share of its shabad
    lines (a table of shrutis with a number on every line is not one), or
    when the page names itself ਤਤਕਰਾ and something matched.
    """
    shabads = sum(1 for e in index.get("entries", []) if e["kind"] == "shabad")
    if not shabads or matched < min_matched:
        return False
    return matched >= min_share * shabads or (index.get("titled") and matched >= min_matched)


def match_entries(entries: list[dict], corpus_index, match_text, accept: float = 0.8) -> int:
    """Each shabad entry's first line matched against the corpus; sets shabad_id and score. Returns how many matched."""
    n = 0
    for e in entries:
        e.setdefault("shabad_id", None)
        e.setdefault("score", None)
        if e["kind"] != "shabad":
            continue
        try:
            hit = match_text(e["text"], corpus_index)
        except Exception:
            hit = None
        if hit and float(hit.get("score") or 0) >= accept:
            e["shabad_id"], e["score"] = hit["shabad_id"], round(float(hit["score"]), 3)
            n += 1
    return n


def by_number(entries: list[dict]) -> dict[int, dict]:
    """{notation number: entry} for the shabad entries the book numbers itself."""
    out: dict[int, dict] = {}
    for e in entries:
        if e["kind"] == "shabad" and e.get("number") is not None:
            out.setdefault(e["number"], e)
    return out


def raag_starts(entries: list[dict]) -> list[dict]:
    """[{key, text, page_printed, number}] for the sections of the index that name a raag."""
    return [{"key": e["raag_key"], "text": e["text"], "page_printed": e["page_printed"], "number": e.get("number")}
            for e in entries if e["kind"] == "section" and e.get("raag_key")]


def page_offset(entries: list[dict], headers: dict[int, int], records: list[dict] | None = None) -> dict:
    """
    Printed page -> scan page as a constant offset, voted by two witnesses:
    the page numbers the running headers carry ({scan page: printed}) and
    the index entries whose shabad a record holds (the record's first
    page against the entry's printed page). {"offset": int|None, "votes":
    {offset: count}, "witnesses": {"headers": n, "entries": n}}; two
    offsets with real support each are reported as `split`, not chosen.
    """
    votes: dict[int, int] = {}
    n_h = n_e = 0
    for scan, printed in (headers or {}).items():
        votes[scan - printed] = votes.get(scan - printed, 0) + 1
        n_h += 1
    by_sid: dict[int, list[int]] = {}
    for r in records or []:
        sid = (r.get("shabad") or {}).get("shabad_id")
        if sid is not None and r.get("pages"):
            by_sid.setdefault(sid, []).append(r["pages"][0])
    for e in entries:
        if e["kind"] != "shabad" or e.get("shabad_id") is None or not by_sid.get(e["shabad_id"]):
            continue
        scan = min(by_sid[e["shabad_id"]], key=lambda p: abs(p - e["page_printed"]))
        votes[scan - e["page_printed"]] = votes.get(scan - e["page_printed"], 0) + 1
        n_e += 1
    if not votes:
        return {"offset": None, "votes": {}, "witnesses": {"headers": n_h, "entries": n_e}, "split": False}
    best = max(votes, key=votes.get)
    others = [v for k, v in votes.items() if k != best and abs(k - best) > 1]
    split = bool(others) and max(others) >= max(3, votes[best] // 2)
    return {"offset": best if not split else None, "votes": {str(k): v for k, v in sorted(votes.items())},
            "witnesses": {"headers": n_h, "entries": n_e}, "split": split}


def missing_entries(entries: list[dict], records: list[dict], offset: int | None, slack: int = 2) -> list[dict]:
    """The shabad entries whose shabad no record holds near the scan page the offset points at: the coverage list."""
    if offset is None:
        return []
    by_sid: dict[int, list[int]] = {}
    for r in records:
        sid = (r.get("shabad") or {}).get("shabad_id")
        if sid is not None:
            by_sid.setdefault(sid, []).extend(r.get("pages") or [])
    out = []
    for e in entries:
        if e["kind"] != "shabad" or e.get("shabad_id") is None:
            continue
        expected = e["page_printed"] + offset
        if not any(abs(p - expected) <= slack for p in by_sid.get(e["shabad_id"], [])):
            out.append({"text": e["text"], "shabad_id": e["shabad_id"], "page_printed": e["page_printed"], "expected_scan_page": expected})
    return out
