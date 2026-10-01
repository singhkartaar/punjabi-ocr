"""
From recognised lines with boxes to the paragraph shape the writings pipeline reads.

lib/writings_pdf.py already knows how to turn positioned lines into
paragraphs: group_lines() by y, left_margin() for the indent a new paragraph
announces itself with, paragraphs() for the breaks (a style flip, a size
change, a wider gap, an indent). Those rules were calibrated in PDF points
on the essays and hold for a scanned book once the boxes are put in the same
units, so this module converts and hands over rather than re-deriving them.

Two things the essays got from the PDF have to come from ink here:

  bold      Gurbani is set bold in every Punjabi sample; italic shear played
            this role for the essays. Bold is read from stroke width: the mean
            of the distance transform over the ink skeleton of a line's box,
            which is robust to the line's length and to the page's darkness.
            A page's widths are bimodal when it carries both weights; Otsu's
            threshold splits them, and is accepted only when the two modes are
            BOLD_RATIO apart. A page without bold text has no split and no
            line is bold. Matched corpus text is the other bold signal, and
            the stronger one: a line found in the corpus is Gurbani whatever
            its stroke says.
  columns   per page, from lib/ocr_zones.page_columns(); the left column is
            read before the right, as read_pdf() does with its bands.
"""
from __future__ import annotations
import statistics

from lib.writings_pdf import group_lines, left_margin, paragraphs

POINTS_PER_PX = 72.0 / 300.0
# Measured on Santhya p.30 at 300 dpi: regular lines 4.04-4.20 px, bold verse
# lines 4.59-4.68, a partly bold line 4.45. The modes are close (the bold face
# is a heavier cut of the same design, not a different width class), so the
# ratio between them is small and the within-mode spread is what makes the
# split safe: 0.16 px inside regular, 0.4 px between the modes.
BOLD_RATIO = 1.06             # the bold mode must be this much wider than the regular one
MIN_WEIGHT_LINES = 3      # lines of each weight a page needs before its split is believed
MIN_INK = 30                  # px of ink for a stroke estimate to mean anything
SIZE_BAND = 0.12              # an OCR line box within this of the page's body height IS the body height:
                              # boxes vary with ascenders and descenders, and the paragraph rules,
                              # written for a PDF's exact font sizes, broke a paragraph at every line


def stroke_width(img, bbox: list) -> float | None:
    """Mean stroke width in px of the ink inside bbox, or None for an empty box."""
    import cv2
    import numpy as np
    x0, y0, x1, y1 = [int(v) for v in bbox]
    crop = img[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if int((ink > 0).sum()) < MIN_INK:
        return None
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    # the skeleton is where the distance is a local maximum; a cheap proxy is
    # the ridge left by a one-pixel erosion
    eroded = cv2.erode(ink, np.ones((3, 3), np.uint8))
    ridge = (ink > 0) & (eroded == 0)
    core = dist[(ink > 0) & ~ridge]
    vals = core if core.size else dist[ink > 0]
    return float(2.0 * vals.mean()) if vals.size else None


def bold_split(widths: list[float]) -> float | None:
    """The stroke width above which a line is bold, or None (one weight only)."""
    ws = sorted(w for w in widths if w)
    if len(ws) < 4:
        return None
    best, best_var = None, None
    for i in range(2, len(ws) - 1):
        a, b = ws[:i], ws[i:]
        var = len(a) * statistics.pvariance(a) + len(b) * statistics.pvariance(b)
        if best_var is None or var < best_var:
            best, best_var = i, var
    lo, hi = ws[:best], ws[best:]
    # two weights need a few lines of each: on a 100 dpi scan (Ten Masters)
    # two faint lines made the "regular" group and the other 34 came out bold,
    # and half the book's prose was then set aside as quotation
    if len(lo) < MIN_WEIGHT_LINES or len(hi) < MIN_WEIGHT_LINES:
        return None
    if statistics.mean(hi) < BOLD_RATIO * statistics.mean(lo):
        return None
    # No test for a gap between the modes: the Santhya's real two weights are
    # as close as a grey scan's noise (measured over 442 pages, the gap
    # against the modes' own spread ran from 0.02 to 3.8, and a rule at 1.0
    # lost 1,400 of its verse lines). What a bold line MEANS is decided
    # downstream, from what the merge made of it (as_runs).
    return (max(lo) + min(hi)) / 2.0


def as_runs(lines: list[dict], page_h: int, scale: float = POINTS_PER_PX) -> list[dict]:
    """
    Merged lines -> the run dicts writings_pdf.group_lines expects:
    {"x", "y", "size", "italic", "text"} in points, y upward. "italic" carries
    what "bold or corpus-matched" means here: a quoted verse.
    """
    out = []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        # A quoted verse is what the merge classified as one (matched, or bold
        # with a verse mark) or as a heading (short and bold). A long bold line
        # the merge left as commentary is prose with a heavy face -- on a grey
        # 120 dpi scan half of every page's prose reads "bold" from stroke
        # width alone -- and must not open a quotation or break a paragraph.
        kind = str(ln.get("kind", ""))
        out.append({"x": x0 * scale, "y": (page_h - y1) * scale, "size": (y1 - y0) * scale,
                    "italic": kind.startswith("gurbani") or kind == "heading"
                    or (bool(ln.get("bold")) and not kind),
                    "text": ln.get("text", ""), "_line": ln})
    return out


def _column_paragraphs(lines: list[dict], page_h: int) -> list[dict]:
    """One column's (or band's) lines as paragraphs, each carrying its "lines"."""
    col = as_runs([ln for ln in lines if ln.get("zone", "body") == "body" and ln.get("text", "").strip()], page_h)
    if not col:
        return []
    grouped = group_lines(col)
    # group_lines keeps the run objects; carry our line records along
    for g in grouped:
        g["_lines"] = [r["_line"] for r in g["runs"]]
    sizes = [g["size"] for g in grouped if len(g["text"]) > 20]
    body = statistics.median(sizes) if sizes else (grouped[0]["size"] if grouped else 10.0)
    for g in grouped:
        if abs(g["size"] - body) <= SIZE_BAND * body:
            g["size"] = body
    paras = paragraphs(grouped, body, left_margin(grouped))
    # paragraphs() concatenates text; map paragraphs back to their lines by
    # walking the grouped lines in order and matching accumulated text
    out = []
    gi = 0
    for p in paras:
        members = []
        acc = ""
        while gi < len(grouped) and len(acc) < len(p["text"]):
            acc = (acc + " " + grouped[gi]["text"]).strip()
            members.extend(grouped[gi]["_lines"])
            gi += 1
        p["lines"] = members
        out.append(p)
    return out


def _verse_paragraph(lines: list[dict], page_h: int) -> dict:
    """The verse lines beside one explanation, as one quote paragraph."""
    lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    x0 = min(ln["bbox"][0] for ln in lines) * POINTS_PER_PX
    size = statistics.median((ln["bbox"][3] - ln["bbox"][1]) * POINTS_PER_PX for ln in lines)
    return {"x0": x0, "size": size, "italic": True, "style": "quote",
            "text": " ".join(ln.get("text", "").strip() for ln in lines if ln.get("text", "").strip()),
            "lines": lines, "verse": True}


def page_paragraphs(lines: list[dict], page_h: int, columns: list[tuple] | None,
                    layout: str = "columns", typical_h: float | None = None) -> list[dict]:
    """
    Paragraphs of one page, in reading order. Each paragraph carries "lines":
    the merged line records it was built from, so a quote's line_ids travel
    with it.

    `layout` "columns" reads the left column whole and then the right, as
    every corpus before the paired reader was built; "auto" and
    "paired-columns" read a two-column page in bands (lib/ocr_pairs): prose
    across the page in order, and in a paired band each explanation
    preceded by the verse lines printed beside it, the two sharing a `pair`
    number and the prose carrying those lines as "explains_lines".
    """
    body = [ln for ln in lines if ln.get("zone", "body") == "body" and ln.get("text", "").strip()]
    if not columns or layout == "columns":
        cols = [[ln for ln in body if lo <= (ln["bbox"][0] + ln["bbox"][2]) / 2.0 < hi] for lo, hi in columns] if columns else [body]
        out = []
        for col in cols:
            out.extend(_column_paragraphs(col, page_h))
        return out
    from lib.ocr_pairs import assign_verse, bands, prose_groups, typical_height
    h = typical_h or typical_height(body)
    out = []
    pair = 0
    for band in bands(body, columns, h, force=(layout == "paired-columns")):
        if band["mode"] == "full":
            out.extend(_column_paragraphs(band["lines"], page_h))
        elif band["mode"] == "columns":
            for col in band["cols"]:
                out.extend(_column_paragraphs(col, page_h))
        else:
            prose = []
            for group in prose_groups(band["prose"], band["verse"], h):
                prose.extend(_column_paragraphs(group, page_h))
            beside = assign_verse(band["verse"], prose, h)
            placed = {id(ln) for group in beside for ln in group}
            stray = [ln for ln in band["verse"] if id(ln) not in placed]
            for p, verse_lines in zip(prose, beside):
                pair += 1
                if verse_lines:
                    v = _verse_paragraph(verse_lines, page_h)
                    v["pair"] = pair
                    out.append(v)
                p["pair"] = pair
                p["explains_lines"] = verse_lines
                out.append(p)
            if stray:
                pair += 1
                v = _verse_paragraph(stray, page_h)
                v["pair"] = pair
                out.append(v)
    return out


def quote_meta(par: dict) -> dict:
    """
    line_ids, their range, and the majority shabad/ang/source of a quote
    paragraph's lines.

    `source` is the scripture the ids belong to (BaniDB codes: G the Guru
    Granth Sahib, D Dasam Bani, B Bhai Gurdas), and it travels with the ids
    from here on: a line_id of Dasam Bani is a number in another id space,
    and without the source beside it a reader would open the wrong verse.
    """
    ids, shabads, angs, scores, sources = [], [], [], [], []
    for ln in par.get("lines", []):
        for m in ln.get("matches", []) or ([ln["match"]] if ln.get("match") else []):
            ids.append(m["line_id"])
            shabads.append(m["shabad_id"])
            angs.append(m["ang"])
            scores.append(m["score"])
            sources.append(m.get("source") or "G")
    if not ids:
        return {}
    return {"line_ids": ids, "line_from": min(ids), "line_to": max(ids),
            "shabad_id": statistics.mode(shabads), "ang": statistics.mode(angs), "source": statistics.mode(sources),
            "match_score": round(min(scores), 3), "match_method": "ocr-corpus-match"}
