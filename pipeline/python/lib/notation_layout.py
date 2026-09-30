"""
A notation page, cut into regions; a book's pages, joined into notations.

What a page of a keertan book holds, top to bottom: a heading (the number,
the raag used, the taal, the laya), the shabad's text with a reference to
where it stands in the Granth, then one grid per section -- ਅਸਥਾਈ, ਅੰਤਰਾ --
each a block of row pairs (swaras over syllables) divided by bars, sometimes
a marker row above it, and often a note to the singer at the end (ਬਾਕੀ
ਤੁਕਾਂ ਅੰਤਰੇ ਤੇ ਲਾਓ). A grid runs onto the next page when the page ends, and
a book that prints the shabad AFTER its grids exists (Sangeet Sagar), so the
page is only half the story: link_pages() strings the regions of a book's
pages into notation spans.

The cut is made from the merged OCR lines (22_ocr_merge.py) and the page's
own ink. The words decide the roles -- a heading names a raag or a taal, a
reference names an ang, a section label is one of a few words, a grid row
is mostly swaras, dashes and held marks -- and the ink decides the extents,
because Tesseract's lines over a grid are unreliable while the rows of ink
are not. Nothing here reads a cell; lib/notation_grid.py does that.
"""
from __future__ import annotations
import re

from lib.notation_text import is_heading_like, is_note_like, parse_heading, parse_ref, token_class
from lib.notation_vocab import gurmukhi_digits, marker_kind, section_label

GRID_CLASSES = ("swara", "held", "rest", "marker", "bar", "matra")
_BAR_CHARS = re.compile(r"[|¦│\[\]{}!]|(?<![।])।(?![।])")
_BAR_SPLIT = re.compile(r"[\s|¦│\[\]{}!]+|(?<![।])।(?![।])")
_PAGENO = re.compile(r"^[\s।॥()\-]*[0-9੦-੯]{1,4}[\s।॥()\-]*$")
_GRANTH_HEADING = re.compile(r"(?:ਮਹਲਾ|ਮਹੱਲਾ|ਮਃ|ਮ[:ਃ])\s*[੧-੯1-9]")
_MARKER_GLYPH = re.compile(r"^[x×X+०੦0oO°\[\)\(\]ਨਤੇੜੇ੨੩੪੧2345]{1,3}$")


# ---- lines -----------------------------------------------------------------

def classify_line(line: dict) -> str:
    """
    stamp | pageno | heading | section | ref | note | gurbani | grid | marker | text
    for one merged line. `kind` and `bold` come from the merge; the text
    decides the rest.
    """
    zone = line.get("zone")
    if zone in ("stamp", "pageno") or line.get("dropped"):
        return "stamp" if zone == "stamp" else "pageno"
    text = (line.get("text") or "").strip()
    if not text:
        return "text"
    # Tesseract glues a bar to its neighbour ("ਪ|ਪ", "ਨੁ।ਧੁ"): bars are separators here
    toks = [t for t in _BAR_SPLIT.split(text) if t]
    bar_count = len(_BAR_CHARS.findall(text))
    if _PAGENO.match(text) and len(toks) <= 2:
        return "pageno"
    if section_label(text) and len(toks) <= 3:
        return "section"
    if is_note_like(text):
        return "note"
    ref = parse_ref(text)
    if ref and (ref["ang_from"] or ref["source"] != "G") and len(toks) <= 12 and (text.startswith("(") or text.startswith("[") or ref["span"][0] <= 2):
        return "ref"
    if _GRANTH_HEADING.search(text) and len(toks) <= 8 and "ਤਾਲ" not in text and not text.startswith("("):
        return "gurbani"                      # ਜੈਤਸਰੀ ਮਹਲਾ ੫ ਘਰੁ ੨: the Granth's own heading, the shabad's first line
    if is_heading_like(text):
        return "heading"
    classes = [token_class(t) for t in toks]
    gridish = sum(c in GRID_CLASSES for c in classes) + bar_count
    if toks and all(_MARKER_GLYPH.match(t) or token_class(t) in ("marker", "matra", "bar") for t in toks) and len(toks) <= 20 and gridish + sum(1 for t in toks if _MARKER_GLYPH.match(t)) >= len(toks):
        return "marker"
    held = sum(1 for c in classes if c == "held") + bar_count
    total = len(toks) + bar_count
    if line.get("kind") in ("gurbani", "gurbani-unmatched") or "॥" in text or "।।" in text:
        # a verse -- unless it is a bol row that OCR happened to read as one
        if gridish / max(1, total) < 0.5 and held < 2:
            return "gurbani"
    if toks and gridish / total >= 0.5:
        return "grid"
    # a bol row: short syllables between held marks and bars; a verse of
    # short words has neither
    short = sum(1 for t in toks if len(t.strip("().,;")) <= 3)
    if len(toks) >= 4 and (short + gridish) / total >= 0.7 and held >= 2:
        return "grid"
    # bold alone says nothing: swaras print heavy too. A bold line with a
    # danda is a verse (a heading of the shabad, ਗਉੜੀ ਮਹਲਾ ੫ ॥)
    if line.get("bold") and len(toks) >= 3 and "।" in text:
        return "gurbani"
    return "text"


def _bbox_union(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ---- the page --------------------------------------------------------------

def page_layout(merged_lines: list[dict], page_w: int, page_h: int, style: dict, page: int,
                ink=None, body_h: int | None = None) -> dict:
    """
    {"page", "kind": notation|prose|blank, "regions": [Region, ...]} in
    reading order, a Region being {"role": heading|shabad|ref|section|
    marker|grid|note|text, "bbox", "lines": [merged line numbers], "text",
    "parsed": {...} for a heading/ref/section}. A grid region's bbox is the
    ink's extent between the lines that bound it when `ink` is given.
    """
    lines = [l for l in merged_lines if l.get("bbox")]
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    roles = [(l, classify_line(l)) for l in lines]
    body_h = body_h or _median([l["bbox"][3] - l["bbox"][1] for l, r in roles if r in ("text", "gurbani", "heading")] or [40])

    regions: list[dict] = []
    cur: dict | None = None

    def close():
        nonlocal cur
        if cur:
            regions.append(cur)
            cur = None

    for line, role in roles:
        if role in ("stamp", "pageno"):
            continue
        n, bbox, text = line.get("n"), list(line["bbox"]), (line.get("text") or "").strip()
        if role in ("heading", "section", "ref", "note", "marker"):
            close()
            region = {"role": role, "bbox": bbox, "lines": [n], "text": text}
            if role == "heading":
                region["parsed"] = parse_heading(text)
            elif role == "section":
                sec = section_label(text)
                region["parsed"] = {"kind": sec[0], "n": sec[1]} if sec else None
            elif role == "ref":
                region["parsed"] = parse_ref(text)
            elif role == "marker":
                region["parsed"] = {"marks": _marker_row(text)}
            regions.append(region)
            continue
        if role in ("gurbani", "text"):
            # a verse block; a text line inside one continues it, a text line
            # alone (an alaap word, an OCR shred) stays text
            if cur and cur["role"] == "shabad":
                cur["lines"].append(n); cur["bbox"] = _bbox_union([cur["bbox"], bbox]); cur["text"] += "\n" + text
                cur["kinds"].append(line.get("kind")); cur["merged"].append(line)
            elif cur and cur["role"] == "grid" and role == "text" and len(text.split()) <= 3:
                # a fragment Tesseract cut out of a grid row
                cur["lines"].append(n); cur["bbox"] = _bbox_union([cur["bbox"], bbox])
            else:
                close()
                cur = {"role": "shabad" if role == "gurbani" or _looks_like_verse(text) else "text",
                       "bbox": bbox, "lines": [n], "text": text, "kinds": [line.get("kind")], "merged": [line]}
            continue
        if role == "grid":
            if cur and cur["role"] == "grid":
                cur["lines"].append(n); cur["bbox"] = _bbox_union([cur["bbox"], bbox])
            elif cur and cur["role"] == "text" and len(cur["lines"]) <= 1:
                # a stray fragment before the grid belongs to it
                cur = {"role": "grid", "bbox": _bbox_union([cur["bbox"], bbox]), "lines": cur["lines"] + [n], "text": ""}
            else:
                close()
                cur = {"role": "grid", "bbox": bbox, "lines": [n], "text": ""}
            continue
    close()

    # between a heading and the first section label, marker row or reference
    # lies the shabad's text, whatever the OCR made of its lines: a Kabitt
    # of Bhai Gurdas matches no corpus line and reads as prose
    regions = _shabad_between(regions, body_h)
    regions = _fold_shreds(regions)
    # a shabad block is a verse the corpus knows -- that is the link the
    # notation needs -- or a verse the book set over a printed reference (a
    # Kabitt of Bhai Gurdas, Dasam Bani, which the corpus does not hold).
    # Prose that mentions a line, tabla bols under a taal heading, and OCR
    # noise that happens to hold a danda are text.
    for i, r in enumerate(regions):
        if r["role"] != "shabad":
            continue
        after_ = regions[i + 1] if i + 1 < len(regions) else None
        over_ref = bool(after_ and after_["role"] == "ref" and re.search("[।॥]", r["text"]))
        if not (_verse_block(r) or over_ref):
            r["role"] = "text"
    # a grid of one line that is really a marker row or a stray
    regions = [r for r in regions if not (r["role"] == "grid" and len(r["lines"]) == 1 and r["bbox"][3] - r["bbox"][1] < body_h * 0.6)]
    # grids separated only by fragments merge; the ink decides the extent
    regions = _merge_adjacent_grids(regions, body_h)
    if ink is not None:
        regions = _ink_grids(regions, ink, page_w, page_h, body_h)
    if ink is not None:
        for r in regions:
            if r["role"] == "grid":
                r["bbox"] = _ink_extent(ink, r["bbox"], page_w, page_h, body_h)
    # a page of notation has grid rows (two or more), a marker row or a section
    # label; a page of prose has long lines and none of those, and is skipped
    # by the linker; anything else (a heading and a shabad before a grid on the
    # next page) is text the next page may continue
    signals = any((r["role"] == "grid" and (len(r["lines"]) >= 2 or r.get("from_ink"))) or r["role"] in ("marker", "section") for r in regions)
    verse_lines = {n for r in regions if r["role"] == "shabad" and _verse_block(r) for n in r["lines"]}
    long_lines = sum(1 for l in lines if len((l.get("text") or "").split()) >= 6 and l.get("n") not in verse_lines)
    matched = any(r["role"] == "shabad" and any(m.get("matches") for m in (r.get("merged") or [])) for r in regions)
    if signals:
        kind = "notation"
    elif not regions:
        kind = "blank"
    elif long_lines >= 5 and not matched and not any(r["role"] == "ref" for r in regions):
        # an essay: long lines, none of them a verse the corpus knows, no reference
        kind = "prose"
    else:
        kind = "text"
    return {"page": page, "kind": kind, "regions": regions, "body_h": body_h}


def _shabad_between(regions: list[dict], body_h: int) -> list[dict]:
    out: list[dict] = []
    i = 0
    while i < len(regions):
        r = regions[i]
        out.append(r)
        if r["role"] == "heading" and not (r.get("parsed") or {}).get("section"):
            j = i + 1
            block: list[dict] = []
            while j < len(regions) and regions[j]["role"] in ("text", "shabad", "grid") and not _is_real_grid(regions[j]):
                block.append(regions[j])
                j += 1
            if block and any(b["role"] in ("shabad", "text") for b in block):
                merged = {"role": "shabad", "bbox": _bbox_union([b["bbox"] for b in block]),
                          "lines": [n for b in block for n in b["lines"]],
                          "text": "\n".join(b.get("text", "") for b in block if b.get("text")),
                          "kinds": [k for b in block for k in (b.get("kinds") or [])],
                          "merged": [m for b in block for m in (b.get("merged") or [])]}
                out.append(merged)
                i = j
                continue
        i += 1
    return out


def _fold_shreds(regions: list[dict]) -> list[dict]:
    """
    A grid of one or two OCR lines, or a text line, standing between two
    verse blocks is a line of the verse Tesseract read as something else
    (a shred of a long line, a swara-like syllable): the three become one
    shabad block. Runs until nothing folds.
    """
    changed = True
    while changed and len(regions) >= 3:
        changed = False
        for i in range(1, len(regions) - 1):
            a, b, c = regions[i - 1], regions[i], regions[i + 1]
            small = (b["role"] == "grid" and len(b["lines"]) <= 2 and not b.get("from_ink")) or \
                    (b["role"] == "text" and len(b["lines"]) <= 1)
            if a["role"] == "shabad" and c["role"] == "shabad" and small:
                merged = {"role": "shabad", "bbox": _bbox_union([a["bbox"], b["bbox"], c["bbox"]]),
                          "lines": a["lines"] + b["lines"] + c["lines"],
                          "text": "\n".join(t for t in (a.get("text"), b.get("text"), c.get("text")) if t),
                          "kinds": (a.get("kinds") or []) + (b.get("kinds") or []) + (c.get("kinds") or []),
                          "merged": (a.get("merged") or []) + (b.get("merged") or []) + (c.get("merged") or [])}
                regions = regions[:i - 1] + [merged] + regions[i + 2:]
                changed = True
                break
    return regions


def _verse_block(region: dict) -> bool:
    """
    A verse the corpus matched: two of its lines, or a third of them (an
    essay that quotes one line of a shabad over a page of prose is not the
    shabad), or the Granth's own heading of a shabad with a matched line.
    """
    merged = region.get("merged") or []
    matched = sum(1 for m in merged if m.get("matches"))
    if not matched:
        return False
    n = max(1, len(region.get("lines") or merged))
    return matched >= 2 or matched / n >= 0.34 or bool(_GRANTH_HEADING.search(region.get("text") or ""))


def _is_real_grid(region: dict) -> bool:
    """A grid region of two OCR lines or more (a swar row over a bol row) is a grid; one line under a heading is a shred."""
    return region["role"] == "grid" and (len(region["lines"]) >= 2 or region.get("from_ink"))


def _looks_like_verse(text: str) -> bool:
    return bool(re.search("[।॥]", text)) and len(text.split()) >= 3


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def _marker_row(text: str) -> list[str | None]:
    """'° x ° x' -> ['0', '×', '0', '×']; a token that is no marker is None."""
    out = []
    for t in text.split():
        k = marker_kind(t)
        if k == "sam" or t in ("x", "X", "ਨ", "+"):
            out.append("×")
        elif k == "khali" or t in ("°", "o", "O", "[o)", "০", "(o)"):
            out.append("0")
        elif k == "tali":
            out.append(gurmukhi_digits(t))
        elif t in ("ਤੇ", "ੜੇ", "੩"):
            out.append("3")
        else:
            out.append(None)
    return out


def _ink_grids(regions: list[dict], ink, page_w: int, page_h: int, body_h: int) -> list[dict]:
    """
    A ruled table Tesseract read nothing of leaves a hole in the regions: a
    section label or a heading, then a tall gap full of ink, then the next
    line. The ink in such a gap is a grid region with no OCR lines; the
    reader works from the ink anyway.
    """
    import numpy as np
    edge = int(page_w * 0.04)
    out: list[dict] = []
    ordered = sorted(regions, key=lambda r: r["bbox"][1])
    for i, r in enumerate(ordered):
        out.append(r)
        # only under a section label, or a heading that names a taal: a paragraph of prose has gaps too
        if not (r["role"] == "section" or (r["role"] == "heading" and (r.get("parsed") or {}).get("taal"))):
            continue
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        ya = r["bbox"][3] + 4
        yb = (nxt["bbox"][1] - 4) if nxt else int(page_h * 0.93)
        if yb - ya < 2.5 * body_h:
            continue
        band = ink[ya:yb, edge:page_w - edge]
        if band.size == 0:
            continue
        rows_ink = np.where((band > 0).sum(axis=1) > 2)[0]
        if len(rows_ink) < 1.5 * body_h:
            continue                                  # a real gap, not a table
        cols_ink = np.where((band > 0).sum(axis=0) > 2)[0]
        bbox = [int(edge + cols_ink.min()), int(ya + rows_ink.min()), int(edge + cols_ink.max()) + 1, int(ya + rows_ink.max()) + 1]
        if (bbox[2] - bbox[0]) < 0.3 * page_w:
            continue
        out.append({"role": "grid", "bbox": bbox, "lines": [], "text": "", "kinds": [], "merged": [], "from_ink": True})
    return sorted(out, key=lambda r: r["bbox"][1])


def _merge_adjacent_grids(regions: list[dict], body_h: int) -> list[dict]:
    out: list[dict] = []
    for r in regions:
        if r["role"] == "grid" and out and out[-1]["role"] == "grid" and r["bbox"][1] - out[-1]["bbox"][3] < body_h * 1.5:
            out[-1]["lines"] += r["lines"]
            out[-1]["bbox"] = _bbox_union([out[-1]["bbox"], r["bbox"]])
        elif r["role"] == "text" and out and out[-1]["role"] == "grid" and len(r["lines"]) <= 2 and len(r["text"].split()) <= 4:
            out[-1]["lines"] += r["lines"]
            out[-1]["bbox"] = _bbox_union([out[-1]["bbox"], r["bbox"]])
        else:
            out.append(r)
    # a grid followed by text fragments then grid again: merge across
    merged: list[dict] = []
    for r in out:
        if r["role"] == "grid" and len(merged) >= 2 and merged[-1]["role"] == "text" and merged[-2]["role"] == "grid" \
                and len(merged[-1]["lines"]) <= 2 and r["bbox"][1] - merged[-2]["bbox"][3] < body_h * 4:
            frag = merged.pop()
            merged[-1]["lines"] += frag["lines"] + r["lines"]
            merged[-1]["bbox"] = _bbox_union([merged[-1]["bbox"], frag["bbox"], r["bbox"]])
        else:
            merged.append(r)
    return merged


def _ink_extent(ink, bbox: list[int], page_w: int, page_h: int, body_h: int) -> list[int]:
    """The grid's real box: the OCR lines' box grown to the ink rows and columns it touches."""
    import numpy as np
    x0, y0, x1, y1 = bbox
    pad_y = int(body_h * 0.6)
    ya, yb = max(0, y0 - pad_y), min(page_h, y1 + pad_y)
    band = ink[ya:yb, :]
    if band.size == 0:
        return bbox
    cols = np.where((band > 0).sum(axis=0) > 0)[0]
    rows = np.where((band > 0).sum(axis=1) > 0)[0]
    if len(cols) == 0 or len(rows) == 0:
        return bbox
    # ignore ink within 3% of the page edges (scanner shadow)
    edge = int(page_w * 0.03)
    cols = cols[(cols > edge) & (cols < page_w - edge)]
    if len(cols) == 0:
        return bbox
    return [int(cols.min()), int(ya + rows.min()), int(cols.max()) + 1, int(ya + rows.max()) + 1]


# ---- the book ---------------------------------------------------------------

def _running_header(region: dict, page_h: int) -> bool:
    """'ਰਾਗ ਜੈਤਸਰੀ' alone at the top of the page: the book's running header, not a notation's heading."""
    p = region.get("parsed") or {}
    if p.get("taal") or p.get("number") or p.get("section"):
        return False
    top = page_h and region["bbox"][1] < 0.09 * page_h
    return bool(top and p.get("raag") and len((region.get("text") or "").split()) <= 3)


def _shabad_heading(region: dict) -> bool:
    """'ਜੈਤਸਰੀ ਮਹਲਾ ੫ ॥': the Granth's own heading of the shabad, part of its text."""
    t = (region.get("text") or "")
    return bool(_GRANTH_HEADING.search(t)) and not re.search("ਤਾਲ", t)


def raag_descriptions(layouts: list[dict]) -> list[dict]:
    """
    The prose a book prints about a raag -- its heading names the raag and
    no taal, and what follows is text until the next heading, section label
    or grid. [{"raag": parsed raag, "heading": region, "page", "pages",
    "regions": [(page, region)]}]. A description may run over a page turn;
    it ends at the first notation signal.
    """
    out: list[dict] = []
    cur: dict | None = None
    prev_page = None
    for lay in layouts:
        page = lay["page"]
        if prev_page is not None and page != prev_page + 1 and cur:
            out.append(cur); cur = None
        prev_page = page
        for r in lay["regions"]:
            role = r["role"]
            p = r.get("parsed") or {}
            if role == "heading" and p.get("raag") and not p.get("taal") and not p.get("section") and not _running_header(r, lay.get("page_h") or 0) \
                    and not _shabad_heading(r):
                if cur and (cur["raag"] or {}).get("key") and (cur["raag"] or {}).get("key") == p["raag"].get("key"):
                    # a sentence of the same description that names the raag again: not a new one
                    if page not in cur["pages"]:
                        cur["pages"].append(page)
                    cur["regions"].append((page, r))
                    continue
                if cur:
                    out.append(cur)
                cur = {"raag": p["raag"], "heading": r, "page": page, "pages": [page], "regions": []}
                continue
            if cur is None:
                continue
            if role in ("text", "shabad", "note", "ref"):
                if page not in cur["pages"]:
                    cur["pages"].append(page)
                cur["regions"].append((page, r))
            else:
                out.append(cur); cur = None
    if cur:
        out.append(cur)
    # a heading with nothing under it is a running header of another kind, not a description
    return [d for d in out if sum(len(r.get("lines") or []) for _, r in d["regions"]) >= 2]


def _label_sections(span: dict) -> None:
    """
    A section with no printed label: the first of a span is the sthai (a
    marker row or a grid opens the notation straight after the shabad, or a
    page turn separated the label from its grid); a later one stays unknown.
    """
    for k, sec in enumerate(span["sections"]):
        if sec.get("kind"):
            continue
        sec["assumed"] = True
        if k == 0:
            sec["kind"], sec["n"] = "sthai", 1


MAX_SPAN_PAGES = 3          # a shabad's notation runs over two or three pages at most; more is the next shabad unread
CONTENT_ROLES = ("grid", "section", "marker", "note", "text")     # what makes a shabad's span a notation
NOTATION_ROLES = ("grid", "section", "marker")                   # what says the grids have begun


def _items(layouts: list[dict]) -> list[dict]:
    """
    Every region of the pages read, in reading order, as {"page", "role",
    "region", "first"} -- `first` marks the first region of a page. A gap in
    the pages read and a page of prose (an essay on a raag, a preface) are
    breaks: {"role": "break"}; nothing links across them. The running
    header is dropped and the Granth's own heading of a shabad (ਗਉੜੀ ਮਹਲਾ ੫)
    is part of the shabad.
    """
    items: list[dict] = []
    repeated = _repeated_tops(layouts)
    prev = None
    for lay in layouts:
        page = lay["page"]
        page_h = lay.get("page_h") or 0
        if (prev is not None and page != prev + 1) or lay.get("kind") == "prose":
            items.append({"page": page, "role": "break", "region": None, "first": True})
        prev = page
        if lay.get("kind") == "prose":
            continue
        first = True
        for r in lay["regions"]:
            role = r["role"]
            if role == "heading" and _running_header(r, page_h):
                continue
            if first and role in ("heading", "text") and _top_key(r, page_h) in repeated:
                continue                                   # "424. ਗੁਰੂ ਅਰਜਨ ਦੇਵ ਰਾਗ ਰਤਨਾਵਲੀ": the book's running header
            if role == "heading" and _shabad_heading(r):
                role = "shabad"
                r = {**r, "role": "shabad", "kinds": ["heading"], "merged": r.get("merged") or []}
            items.append({"page": page, "role": role, "region": r, "first": first})
            first = False
    return items


_AT_NUMBER = re.compile("(?:ਸ਼ਬਦ|ਸਬਦ)\\s*ਨੰ")
_DIGITS_RE = re.compile(r"[0-9੦-੯.:।()\-]+")


def _top_key(region: dict, page_h: int) -> str | None:
    """The words of a one-line region in the top tenth of the page, digits dropped; None lower down."""
    if not page_h or region["bbox"][1] > 0.1 * page_h or len(region.get("lines") or []) > 1:
        return None
    words = _DIGITS_RE.sub(" ", region.get("text") or "").split()
    return " ".join(words[:2]) if len(words) >= 2 else None


def _repeated_tops(layouts: list[dict]) -> set[str]:
    """The top-of-page texts that recur on three pages or more (or on most of a short read): running headers."""
    seen: dict[str, int] = {}
    n = 0
    for lay in layouts:
        n += 1
        regs = lay.get("regions") or []
        if regs:
            key = _top_key(regs[0], lay.get("page_h") or 0)
            if key:
                seen[key] = seen.get(key, 0) + 1
    need = 3 if n >= 6 else 2
    return {k for k, c in seen.items() if c >= need}


def _sid_of(regions: list[dict]) -> int | None:
    """The shabad the corpus matches on these regions' lines vote for, or None."""
    from lib.notation_resolve import votes_from_lines
    weight, _, _ = votes_from_lines([m for r in regions for m in (r.get("merged") or [])])
    return max(weight, key=weight.get) if weight else None


def _named_heading(region: dict) -> dict | None:
    """The parse of a heading that names a raag or a taal (a notation's heading, not a section label)."""
    p = region.get("parsed") or {}
    return p if (p.get("raag") or p.get("taal")) and not p.get("section") else None


def _is_lead(item: dict) -> bool:
    """A heading, or a one-line scrap (a number, a shred), standing between one notation and the next shabad leads the next in."""
    if item["role"] == "heading":
        return True
    return item["role"] == "text" and len(item["region"].get("lines") or []) <= 1


def _segments(items: list[dict], after: bool) -> list[dict]:
    """
    The items cut into notations. A notation is a shabad and what the book
    prints after it (before it, in a book that sets the shabad after the
    grids) up to the next shabad: the next shabad's lead-in -- its heading,
    its number -- goes with the next; a shabad the corpus links to the same
    shabad as the open notation (a verse printed before each of its grids)
    continues it. A heading that names a raag or a taal after a finished
    notation opens the next; a raag heading followed by prose (a
    description) leaves the description to be dropped later, shabadless.
    """
    segs: list[dict] = []
    cur: dict | None = None

    def close():
        nonlocal cur
        if cur and cur["items"]:
            segs.append(cur)
        cur = None

    def open_(**kw):
        return {"items": [], "sid": None, "after_break": False, "opened_by": None, **kw}

    def has(seg, roles):
        return any(it["role"] in roles for it in seg["items"])

    i = 0
    broke = True
    while i < len(items):
        it = items[i]
        role = it["role"]
        if role == "break":
            close()
            broke = True
            i += 1
            continue
        if cur is None:
            cur = open_(after_break=broke)
            broke = False
        if role == "shabad":
            group = []
            j = i
            while j < len(items) and items[j]["role"] in ("shabad", "ref"):
                group.append(items[j])
                j += 1
            sid = _sid_of([g["region"] for g in group if g["role"] == "shabad"])
            if after:
                cur["items"].extend(group)
                cur["sid"] = cur["sid"] or sid
                close()
                i = j
                continue
            same = sid is not None and sid == cur["sid"]
            if (has(cur, ("shabad",)) and not same) or (not has(cur, ("shabad",)) and has(cur, NOTATION_ROLES)):
                # the next shabad -- or the first, after grids whose shabad was not read (the previous page, a failed read)
                lead: list[dict] = []
                while cur["items"] and _is_lead(cur["items"][-1]):
                    lead.insert(0, cur["items"].pop())
                close()
                cur = open_(items=lead, opened_by="shabad")
            cur["items"].extend(group)
            if cur["sid"] is None:
                cur["sid"] = sid
            i = j
            continue
        if role == "heading":
            p = _named_heading(it["region"])
            if p:
                done = has(cur, ("shabad",)) and has(cur, NOTATION_ROLES)
                raag_only = p.get("raag") and not p.get("taal") and not p.get("number")
                if after or done or (raag_only and has(cur, NOTATION_ROLES)):
                    close()
                    cur = open_(opened_by="raag" if raag_only else "heading")
            cur["items"].append(it)
            i += 1
            continue
        cur["items"].append(it)
        i += 1
    close()
    return segs


def _cap_pages(seg: dict, dropped: list[dict]) -> None:
    """A notation runs over MAX_SPAN_PAGES pages at most from its shabad's page; the rest is the next one, unread."""
    pages = sorted({it["page"] for it in seg["items"]})
    start = min((it["page"] for it in seg["items"] if it["role"] == "shabad"), default=pages[0])
    keep = [p for p in pages if p < start + MAX_SPAN_PAGES]
    if len(keep) == len(pages):
        return
    cut = [it for it in seg["items"] if it["page"] not in keep]
    seg["items"] = [it for it in seg["items"] if it["page"] in keep]
    seg["capped"] = True
    dropped.append({"why": "beyond-cap", "pages": sorted({it["page"] for it in cut}),
                    "roles": [it["role"] for it in cut]})


def _span_from_items(seg: dict) -> dict:
    """The span dict the parser reads, from a segment's items."""
    span = {"heading": None, "heading_page": None, "headings": [], "shabad": [], "ref": None, "sections": [],
            "notes": [], "text": [], "pages": [], "continued": False, "extent": {},
            "sid": seg.get("sid"), "inherited": bool(seg.get("inherited")), "capped": bool(seg.get("capped"))}

    def touch(page):
        if page not in span["pages"]:
            span["pages"].append(page)

    def section(page):
        if not span["sections"]:
            span["sections"].append({"label": None, "kind": None, "n": None, "page": page, "grids": [], "markers": [], "taal": None})
        return span["sections"][-1]

    for it in seg["items"]:
        page, r, role = it["page"], it["region"], it["role"]
        touch(page)
        ext = span["extent"].get(page)
        span["extent"][page] = _bbox_union([ext, r["bbox"]]) if ext else list(r["bbox"])
        if role == "heading":
            p = r.get("parsed") or {}
            if p.get("section"):
                sec = p["section"]
                span["sections"].append({"label": r, "kind": sec.get("kind"), "n": sec.get("n"), "page": page,
                                         "grids": [], "markers": [], "taal": p.get("taal")})
            else:
                span["headings"].append((page, r))
                if span["heading"] is None:
                    span["heading"], span["heading_page"] = r, page
        elif role == "section":
            sec = r.get("parsed") or {}
            span["sections"].append({"label": r, "kind": sec.get("kind"), "n": sec.get("n"), "page": page,
                                     "grids": [], "markers": [], "taal": None})
        elif role == "marker":
            section(page)["markers"].append((page, r))
        elif role == "grid":
            section(page)["grids"].append((page, r))
        elif role == "shabad":
            span["shabad"].append((page, r))
        elif role == "ref":
            if span["ref"] is None:
                span["ref"] = r
        elif role == "note":
            span["notes"].append((page, r))
        elif role == "text":
            span["text"].append((page, r))
    # the heading's facts may stand on two lines (the raag, then "ਤਾਲ ਤੀਨਤਾਲ"): one heading, filled from all
    if len(span["headings"]) > 1:
        first = span["heading"]
        parsed = dict(first.get("parsed") or {})
        for _, r in span["headings"][1:]:
            for k, v in (r.get("parsed") or {}).items():
                if v and not parsed.get(k):
                    parsed[k] = v
        span["heading"] = {**first, "parsed": parsed, "text": " / ".join(r["text"] for _, r in span["headings"])}
    first_item = seg["items"][0]
    span["continued"] = bool(seg.get("after_break") is False and first_item["first"]
                             and first_item["role"] in ("grid", "marker", "section")) or bool(seg.get("continued"))
    _label_sections(span)
    return span


def link_pages(layouts: list[dict], style: dict, dropped: list[dict] | None = None) -> list[dict]:
    """
    Regions of consecutive pages, strung into notation spans, one per shabad:
    {"heading": region|None, "heading_page", "headings": [(page, region)], "shabad": [(page, region)],
     "ref": region|None, "sections": [{"label", "kind", "n", "page", "grids", "markers", "taal"}],
     "notes": [(page, region)], "text": [(page, region)], "pages": [..], "extent": {page: bbox},
     "continued": bool, "inherited": bool, "capped": bool}

    The shabad is the anchor. A notation is its shabad and everything the
    book prints after it -- grids, taans, tihais, notes, a notation set as
    text -- up to the next shabad, a raag's heading, a page of prose or the
    end of the pages read, and over MAX_SPAN_PAGES pages at most. What has
    no shabad (a preface, an exercise, a raag description) is dropped and
    listed in `dropped`; a notation opened by a heading straight after
    another (the same shabad in a second taal) inherits that one's shabad.
    Where the book sets the shabad after its grids (style.shabad_position
    == "after") the shabad closes the notation instead of opening it.

    A window of pages cuts notations at its edges: a shabad on the last
    page read with nothing after it, and grids on the first page read with
    no shabad before them, are kept as partial notations.
    """
    after = style.get("shabad_position") == "after"
    items = _items(layouts)
    if not items:
        return []
    pages_read = sorted({it["page"] for it in items if it["role"] != "break"})
    first_read, last_read = (pages_read[0], pages_read[-1]) if pages_read else (None, None)
    segs = _segments(items, after)
    out: list[dict] = []
    dropped = dropped if dropped is not None else []
    prev: dict | None = None
    for seg in segs:
        roles = {it["role"] for it in seg["items"]}
        pages = sorted({it["page"] for it in seg["items"]})
        has_shabad = "shabad" in roles
        has_content = bool(roles & set(CONTENT_ROLES))
        opener = seg["items"][0]["region"] if seg["items"] and seg["items"][0]["role"] == "heading" else None
        numbered = bool(opener and (opener.get("parsed") or {}).get("number") is not None)
        if not has_shabad and has_content and seg.get("opened_by") == "heading" and not numbered and prev is not None \
                and prev.get("sid") is not None and not seg.get("after_break") \
                and any(it["role"] == "shabad" for it in prev["items"]):
            # the same shabad set again in another taal, its text not reprinted
            seg["items"] = [it for it in prev["items"] if it["role"] in ("shabad", "ref")] + seg["items"]
            seg["sid"], seg["inherited"] = prev["sid"], True
            has_shabad = True
        edge_shabad = has_shabad and not has_content and (last_read in pages if not after else first_read in pages)
        edge_grid = has_content and not has_shabad and (first_read in pages if not after else last_read in pages)
        # a notation the book numbers, with the raag and the taal in its heading, is one
        # even with no shabad text under it ("ਇਹ ਸ਼ਬਦ ਨੰ: ੩ ਤੇ ਲਿਖਿਆ ਹੈ"): the parser
        # borrows the shabad by that note, or reads it off the bol row. A numbered
        # exercise under a taal alone is not.
        parsed = (opener or {}).get("parsed") or {}
        at_number = any(it["role"] == "note" and _AT_NUMBER.search(it["region"].get("text") or "") for it in seg["items"])
        counted = has_content and not has_shabad and numbered and "grid" in roles \
            and ((parsed.get("raag") and parsed.get("taal")) or at_number)
        if not (has_shabad and has_content) and not edge_shabad and not edge_grid and not counted:
            dropped.append({"why": "no-shabad" if not has_shabad else "no-notation", "pages": pages,
                            "roles": sorted(roles), "opened_by": seg.get("opened_by")})
            prev = seg if has_shabad else prev
            continue
        if edge_grid:
            seg["continued"] = True
        if not after:
            _cap_pages(seg, dropped)
        out.append(_span_from_items(seg))
        prev = seg
    return out
