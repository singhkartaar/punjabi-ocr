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
    # a shabad block of one short line with no danda is text, not a verse
    for r in regions:
        if r["role"] == "shabad" and len(r["lines"]) == 1 and "॥" not in r["text"] and "gurbani" not in (r.get("kinds") or []) and len(r["text"].split()) < 4:
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
    long_lines = sum(1 for l in lines if len((l.get("text") or "").split()) >= 6)
    if signals:
        kind = "notation"
    elif not regions:
        kind = "blank"
    elif long_lines >= 5 and not any(r["role"] == "ref" for r in regions):
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


def _is_real_grid(region: dict) -> bool:
    """A grid region of at least three OCR lines is a grid; one or two lines between verses are verse."""
    return region["role"] == "grid" and len(region["lines"]) >= 3


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
    return bool(re.search("ਮਹਲਾ|ਮਹੱਲਾ|ਮਃ|ਮ[:ਃ]\s*[੧-੯1-9]", t)) and not re.search("ਤਾਲ", t)


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


def link_pages(layouts: list[dict], style: dict) -> list[dict]:
    """
    Regions of consecutive pages, strung into notation spans:
    {"heading": region|None, "heading_page", "shabad": [(page, region)], "ref": region|None,
     "sections": [{"label": region|None, "page", "grids": [(page, region)], "markers": [(page, region)]}],
     "notes": [(page, region)], "pages": [..], "continued": bool}
    A heading opens a span; a page that begins with a grid, a section label or
    a marker row continues the open one. Where the book prints the shabad
    after its grids (style.shabad_position == "after"), a shabad block closes
    the span that precedes it instead of opening the next.
    """
    after = style.get("shabad_position") == "after"
    spans: list[dict] = []
    cur: dict | None = None

    def new_span(page):
        return {"heading": None, "heading_page": None, "shabad": [], "ref": None, "sections": [],
                "notes": [], "pages": [page], "continued": False}

    def touch(span, page):
        if page not in span["pages"]:
            span["pages"].append(page)

    def section(span, page):
        if not span["sections"]:
            span["sections"].append({"label": None, "page": page, "grids": [], "markers": []})
        return span["sections"][-1]

    prev_page = None
    for lay in layouts:
        page = lay["page"]
        if lay.get("kind") == "prose":
            continue
        if prev_page is not None and page != prev_page + 1 and cur:
            spans.append(cur)                      # a gap in the pages read: nothing continues across it
            cur = None
        prev_page = page
        first = True
        page_h = lay.get("page_h") or 0
        for r in lay["regions"]:
            role = r["role"]
            if role == "heading" and _running_header(r, page_h):
                continue
            if role == "heading" and _shabad_heading(r):
                role = "shabad"
                r = {**r, "role": "shabad", "kinds": ["heading"], "merged": r.get("merged") or []}
            if role == "heading" and (r["parsed"].get("raag") or r["parsed"].get("taal")) and not r["parsed"].get("section"):
                if cur and (after and not cur["shabad"]):
                    # the shabad of the previous span has not come yet; a new heading means it never will
                    pass
                if cur:
                    spans.append(cur)
                cur = new_span(page)
                cur["heading"], cur["heading_page"] = r, page
            elif role in ("section", "heading"):
                if cur is None:
                    cur = new_span(page); cur["continued"] = first
                touch(cur, page)
                sec = r["parsed"] if role == "section" else r["parsed"].get("section")
                if role == "heading" and not r["parsed"].get("section"):
                    sec = None
                cur["sections"].append({"label": r, "kind": (sec or {}).get("kind"), "n": (sec or {}).get("n"),
                                        "page": page, "grids": [], "markers": [],
                                        "taal": r["parsed"].get("taal") if role == "heading" else None})
            elif role == "marker":
                if cur is None:
                    cur = new_span(page); cur["continued"] = first
                touch(cur, page)
                section(cur, page)["markers"].append((page, r))
            elif role == "grid":
                if cur is None or (after and cur["shabad"] and cur["sections"]):
                    if cur:
                        spans.append(cur)          # the shabad came after the grid: this grid is the next notation's
                    cur = new_span(page); cur["continued"] = first and not (after and spans)
                touch(cur, page)
                section(cur, page)["grids"].append((page, r))
            elif role == "shabad":
                if after:
                    if cur is None:
                        cur = new_span(page); cur["continued"] = first
                    touch(cur, page)
                    cur["shabad"].append((page, r))
                else:
                    if cur is None or (cur["sections"] and cur["shabad"]):
                        # a shabad after grids with no new heading: a new notation without a heading
                        if cur:
                            spans.append(cur)
                        cur = new_span(page)
                    touch(cur, page)
                    cur["shabad"].append((page, r))
            elif role == "ref":
                if cur is None:
                    cur = new_span(page)
                touch(cur, page)
                cur["ref"] = r
            elif role == "note":
                if cur:
                    touch(cur, page)
                    cur["notes"].append((page, r))
            elif role == "text":
                pass
            first = False
    if cur:
        spans.append(cur)
    for span in spans:
        _label_sections(span)
    return spans
