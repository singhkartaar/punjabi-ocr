"""
Which part of the page a recognised line belongs to.

A book page is not only its text. The Santhya carries a running header --
"ਸੋਦਰ ਆਸਾ ਮ:੧   ( ੧੮੩ )   (ਰਹਿਰਾਸ-ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ ੮-੯" -- whose ang range is the
single most useful fact on the page, because it says which forty lines of the
corpus the bold Gurbani below can be. It also carries a footnote block under a
rule, and a "Page 201 of 530" stamp the scanner added. The Sahib Singh books
carry "Page 61      www.sikhbookclub.com". None of that is the author's text
and all of it would end up inside a paragraph if it were not set aside first.

Zones are decided from geometry and a few regexes, on the engine's line boxes:

  header    within the top HEADER_BAND of the page and short
  stamp     the scanner's text, matched by regex or by overlap with the box the
            PDF's own text layer gave for it (lib/ocr_pages.page_stamps)
  footnote  below the horizontal rule when there is one, else a trailing block
            of lines set smaller than the body that starts with a digit
  pageno    a line that is only a number, at the top or the bottom
  body      everything else

An engine that emits layout classes of its own (dots.ocr, IndicOCR) overrides
the geometry where it is confident; Tesseract and Vision have none.

Columns are found per PAGE, not per document as lib/writings_pdf.columns()
does: the Santhya mixes single- and two-column pages. The method is the same
-- a quiet band in the profile of where words BEGIN -- but the gutter in the
Santhya is a hairline rule with ~55 px of white either side at 300 dpi, far
narrower than the essays' 80-point gutters, so the width threshold is relative
to the page rather than the essays' absolute constant.
"""
from __future__ import annotations
import re

HEADER_BAND = 0.075        # share of page height
FOOTER_BAND = 0.06
HEADER_MAX_CHARS = 90
FOOTNOTE_SMALL = 0.85      # footnote line height / body line height
RULE_INK = 0.25            # a row this dark across the page is a rule
RULE_TOP = 0.55            # rules are looked for in the bottom 45% ...
RULE_BOTTOM = 0.94         # ... but above the stamp
# Measured at 300 dpi: the Santhya's gutter is a hairline rule with ~55 px of
# white either side (2.6% of a 2130 px page); a justified English page shows
# 24 px runs with no word START in them by chance. So the band must be wider
# than 2% of the page AND, when the image is given, carry almost no ink
# between the first and last body line -- a gutter is empty, a coincidence
# in word starts is not.
MIN_GUTTER = 0.02          # share of page width a column gap must span
MIN_GUTTER_INK = 0.012     # share of page width the ink-free run (rule included) must span
INK_CLEAN = 0.2            # a column is white below this share of the page's median column ink
CROSSING_MAX = 0.3         # share of body lines allowed to cross the gutter (Tesseract merges a few)
RULE_MAX = 20              # px of dark columns tolerated inside the run (Santhya p.201: the rule is 15 px with its halo)
_LETTERS = re.compile("[A-Za-z0-9ਅ-ਹਖ਼-ਫ਼੦-੯ऀ-ॿ]{2,}")
COLUMN_SHARE = 0.25        # each column must hold this share of the words
BIN = 8                    # px, resolution of the start-position profile

STAMP = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|www\.\S+|\S*sikhbookclub\S*)\s*$", re.I)
STAMP_WORD = re.compile(r"sikhbookclub|^page\s+\d+", re.I)
DIGITS = "0-9੦-੯"
PAGENO = re.compile("^[\\s()%s।]+$" % DIGITS)
FOOTNOTE_START = re.compile("^[%s]{1,2}[.)]?\\s" % DIGITS)
# The Santhya header: "<section> ( <book page> ) (<bani>-ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ <ang>-<ang>"
BOOK_PAGE = re.compile("\\(\\s*([%s]+)\\s*\\)" % DIGITS)
ANG_RANGE = re.compile("ਪੰਨਾ\\s*([%s]+)\\s*(?:[-–—]\\s*([%s]+))?" % (DIGITS, DIGITS))
TRAILING_NUMBER = re.compile("([%s]+)\\s*$" % DIGITS)
GURMUKHI_DIGITS = "੦੧੨੩੪੫੬੭੮੯"


def to_int(s: str) -> int | None:
    """'੧੮੩' or '183' -> 183."""
    digits = "".join(str(GURMUKHI_DIGITS.index(c)) if c in GURMUKHI_DIGITS else c
                     for c in s if c.isdigit() or c in GURMUKHI_DIGITS)
    return int(digits) if digits else None


def header_hints(text: str) -> dict:
    """
    {"ang_from", "ang_to", "book_page", "section"} from a running header;
    every key present, None where the header does not say.
    """
    out = {"ang_from": None, "ang_to": None, "book_page": None, "section": None}
    if not text:
        return out
    m = ANG_RANGE.search(text)
    if m:
        out["ang_from"] = to_int(m.group(1))
        out["ang_to"] = to_int(m.group(2)) if m.group(2) else out["ang_from"]
        if out["ang_from"] and out["ang_to"] and out["ang_to"] < out["ang_from"]:
            out["ang_to"] = out["ang_from"]
    m = BOOK_PAGE.search(text)
    if m:
        out["book_page"] = to_int(m.group(1))
    elif not out["ang_from"]:
        m = TRAILING_NUMBER.search(text)
        if m:
            out["book_page"] = to_int(m.group(1))
    head = text.split("(")[0].strip(" -–:")
    if head and not PAGENO.match(head):
        out["section"] = head
    return out


STAMP_ANY = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.I)


def is_stamp(text: str) -> bool:
    """A viewer's "Page N of M" or a site's watermark, also when a few
    specks of OCR noise precede it ("- ਕ - ! Page 8 of 530")."""
    text = text or ""
    if STAMP.match(text) or STAMP_WORD.search(text):
        return True
    return len(text) < 40 and bool(STAMP_ANY.search(text))


def _overlaps(a: list, b: list, frac: float = 0.5) -> bool:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy >= frac * area


def footnote_rule_y(img) -> int | None:
    """Row of the horizontal rule above a footnote block, or None."""
    import numpy as np
    h, w = img.shape[:2]
    band = img[int(h * RULE_TOP):int(h * RULE_BOTTOM)]
    dark = (band < 128).mean(axis=1)
    rows = np.where(dark >= RULE_INK)[0]
    if rows.size == 0:
        return None
    # the lowest run of dark rows is the rule (an underline higher up loses)
    return int(h * RULE_TOP) + int(rows.max())


def classify_zones(lines: list[dict], page_w: int, page_h: int, stamps: list[dict] | None = None,
                   rule_y: int | None = None) -> list[dict]:
    """
    Each line with a "zone" key set: header | stamp | footnote | pageno | body.
    A zone an engine already set (dots.ocr categories) is kept.
    """
    heights = sorted((ln["bbox"][3] - ln["bbox"][1]) for ln in lines if ln.get("text", "").strip())
    body_h = heights[len(heights) // 2] if heights else 0
    out = []
    for ln in lines:
        rec = dict(ln)
        x0, y0, x1, y1 = rec["bbox"]
        text = (rec.get("text") or "").strip()
        yc = (y0 + y1) / 2.0
        zone = rec.get("zone")
        if zone in ("header", "stamp", "footnote", "pageno", "body"):
            pass
        elif not text:
            zone = "body"
        elif is_stamp(text) or any(_overlaps(rec["bbox"], s["bbox"]) for s in (stamps or [])):
            zone = "stamp"
        elif PAGENO.match(text) and (yc < page_h * HEADER_BAND * 1.5 or yc > page_h * (1 - FOOTER_BAND * 1.5)):
            zone = "pageno"
        elif yc < page_h * HEADER_BAND and len(text) <= HEADER_MAX_CHARS:
            zone = "header"
        elif rule_y is not None and y0 > rule_y:
            zone = "footnote"
        else:
            zone = "body"
        rec["zone"] = zone
        out.append(rec)
    # No rule: the trailing run of small body lines, from the first one that
    # opens with a footnote number (continuation lines of a note do not, so
    # the run is gathered first and the number looked for inside it).
    if rule_y is None and body_h:
        trailing: list[dict] = []
        for rec in sorted((r for r in out if r["zone"] == "body"), key=lambda r: -r["bbox"][1]):
            if (rec["bbox"][3] - rec["bbox"][1]) < body_h * FOOTNOTE_SMALL:
                trailing.append(rec)
            else:
                break
        trailing.reverse()                                # top to bottom
        starts = [i for i, r in enumerate(trailing) if FOOTNOTE_START.match(r.get("text", ""))]
        if starts:
            for rec in trailing[starts[0]:]:
                rec["zone"] = "footnote"
    return out


def find_vertical_rule(img, page_w: int | None = None) -> int | None:
    """
    Center column of a hairline vertical divider rule down the page body,
    with clear white margins on either side, or None.
    """
    h, w = img.shape[:2]
    pw = page_w or w
    ink = (img < 180)
    y0, y1 = int(h * 0.15), int(h * 0.85)
    body = ink[y0:y1]
    col_ink = body.mean(axis=0)
    lo, hi = int(pw * 0.20), int(pw * 0.80)
    for x in range(lo, hi):
        if col_ink[x] > 0.20:
            rule_w = 1
            while x + rule_w < hi and col_ink[x + rule_w] > 0.15:
                rule_w += 1
            if 1 <= rule_w <= 25:
                left_gap = col_ink[max(0, x - 25):max(0, x - 2)].min()
                right_gap = col_ink[min(pw - 1, x + rule_w + 2):min(pw - 1, x + rule_w + 25)].min()
                if left_gap < 0.05 and right_gap < 0.05:
                    return x + rule_w // 2
    return None


def columns_by_ink(img, words: list[dict], page_w: int, lines: list[dict] | None = None) -> list[tuple[int, int]]:
    """
    The gutter read off the page image: the widest run of (nearly) ink-free
    pixel columns down the body, a hairline rule allowed inside it, "ink-free"
    being relative to the page's own density (a light JPEG scan of English
    prose has faint columns everywhere). The candidate is then checked
    against the LINE boxes: on a two-column page almost no line crosses the
    gutter; on a single-column page every line does.
    """
    import numpy as np
    rule = find_vertical_rule(img, page_w)
    if rule is not None:
        if lines:
            body_lines = [ln for ln in lines if ln.get("text", "").strip() and ln.get("zone") in (None, "body")]
            crossing = sum(1 for ln in body_lines if ln["bbox"][0] < rule - 20 and ln["bbox"][2] > rule + 20)
            if body_lines and crossing > CROSSING_MAX * len(body_lines):
                return []
        return [(0, rule), (rule, page_w)]

    ys = [w["bbox"][1] for w in words if w.get("text", "").strip()] + \
         [w["bbox"][3] for w in words if w.get("text", "").strip()]
    if len(ys) < 20:
        return []
    y0, y1 = max(0, min(ys)), min(img.shape[0], max(ys))
    body = img[y0:y1]
    col_ink = (body < 128).mean(axis=0)
    lo, hi = int(page_w * 0.2), int(page_w * 0.8)
    median = float(np.median(col_ink[lo:hi])) or 0.05
    clean = max(0.004, INK_CLEAN * median)
    # Measured on Santhya p.201: white at 762-778 (ink 0.003-0.018), the rule
    # at 780-790 (0.3-0.4), white again at 792-798; ~36 px in all, 1.7% of
    # the page. On an English page no column of the body drops below 0.06.
    best, run_from, dark = (0, None), None, 0
    for x in range(lo, hi):
        if col_ink[x] < clean:
            if run_from is None:
                run_from = x
            dark = 0
        else:
            dark += 1
            if run_from is not None and dark > RULE_MAX:      # more than a rule: the run ends
                if x - dark - run_from > best[0]:
                    best = (x - dark - run_from, run_from)
                run_from, dark = None, 0
    if run_from is not None and hi - run_from > best[0]:
        best = (hi - run_from, run_from)
    if best[0] < page_w * MIN_GUTTER_INK:
        return []
    at = best[1] + best[0] // 2
    xs = [w["bbox"][0] for w in words if _LETTERS.search(w.get("text", ""))]
    left = sum(1 for x in xs if x < at)
    if not xs or min(left, len(xs) - left) < COLUMN_SHARE * len(xs):
        return []
    if lines:
        body_lines = [ln for ln in lines if ln.get("text", "").strip() and ln.get("zone") in (None, "body")]
        crossing = sum(1 for ln in body_lines if ln["bbox"][0] < at - 20 and ln["bbox"][2] > at + 20)
        if body_lines and crossing > CROSSING_MAX * len(body_lines):
            return []
    return [(0, at), (at, page_w)]


def page_columns(words: list[dict], page_w: int, img=None, lines: list[dict] | None = None) -> list[tuple[int, int]]:
    """
    [(x_lo, x_hi), ...] for a two-column page, [] for one column.
    With the page image the gutter is read from the ink (columns_by_ink);
    without it, from where words BEGIN, as lib/writings_pdf.columns does.
    """
    if img is not None:
        found = columns_by_ink(img, words, page_w, lines)
        if found:
            return found
        # the word-start profile may still find it (a rule read as ink on a
        # dark scan); the line-crossing test decides either way
        found = page_columns(words, page_w)
        if found and lines:
            at = found[0][1]
            body_lines = [ln for ln in lines if ln.get("text", "").strip() and ln.get("zone") in (None, "body")]
            crossing = sum(1 for ln in body_lines if ln["bbox"][0] < at - 20 and ln["bbox"][2] > at + 20)
            if body_lines and crossing > CROSSING_MAX * len(body_lines):
                return []
        return found
    # the vertical rule of a gutter is read as "|" (or "I", "l", "।") on every
    # line; those tokens BEGIN inside the gutter and would close it
    xs = sorted(w["bbox"][0] for w in words if _LETTERS.search(w.get("text", "")))
    if len(xs) < 40:
        return []
    bins = page_w // BIN + 1
    starts = [0] * bins
    for x in xs:
        starts[min(bins - 1, max(0, x // BIN))] += 1
    # A bin is quiet when nothing starts in it; one stray start per 400 words
    # is tolerated (a rule the engine read as "|" must not close the gutter).
    # A tolerance of 1 on a 100-word page would make every bin quiet.
    limit = len(xs) // 400
    min_bins = max(2, int(page_w * MIN_GUTTER / BIN))
    best, run_from = (0, None), None
    for i, v in enumerate(starts + [10 ** 9]):
        quiet = v <= limit and 0.2 < i / bins < 0.8
        if quiet and run_from is None:
            run_from = i
        elif not quiet and run_from is not None:
            if i - run_from > best[0]:
                best = (i - run_from, (run_from + i) / 2.0)
            run_from = None
    if best[0] < min_bins or best[1] is None:
        return []
    at = int(best[1] * BIN)
    left = sum(1 for x in xs if x < at)
    if min(left, len(xs) - left) < COLUMN_SHARE * len(xs):
        return []
    return [(0, at), (at, page_w)]


def header_of(lines: list[dict]) -> dict:
    """Hints from every header-zone line on a page, first non-empty value wins."""
    out = {"ang_from": None, "ang_to": None, "book_page": None, "section": None}
    for ln in lines:
        if ln.get("zone") != "header":
            continue
        h = header_hints(ln.get("text", ""))
        for k, v in h.items():
            if out[k] is None and v is not None:
                out[k] = v
    return out
