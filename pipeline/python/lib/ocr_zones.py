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
# A hairline rule down the page body is a gutter the printer drew, and is
# believed over everything the engine's lines say: where Tesseract read across
# it, those lines are the ones split_at_gutter is for, so the crossing test
# above must not veto it. Shares of the page, not pixels, so a 200 dpi scan
# and a 400 dpi one are judged alike (measured on the Santhya at 300 dpi: the
# rule inks 30-40% of body rows over 3-15 px, with ~55 px of white either side).
RULE_MAX_W = 0.01          # share of page width: wider than this is a bar, a picture edge or a text column
RULE_MIN_RUN = 0.12        # share of page height one unbroken vertical stroke must span: text in a
                           # column never runs longer than a line, a rule beside a paired band does
RULE_GAP = 0.003           # share of page width of white that must touch the rule on each side
                           # (Santhya p.201: 16 px on the left, 9 px on the right, then the columns'
                           # text; p.208 sets its verse 7 px from the rule)
RULE_GAP_INK = 0.08        # the most ink a column of that white may carry, over the rule's own rows
RULE_BAND = (0.15, 0.85)   # share of page height the rule is looked for in (clear of header and footnotes)
COLUMN_SHARE_RULE = 0.10   # with a rule, each side needs only this share of word starts (verse columns are narrow)

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


def header_hints(text: str, pattern=None) -> dict:
    """
    {"ang_from", "ang_to", "book_page", "section"} from a running header;
    every key present, None where the header does not say.

    `pattern` is a book's own regular expression (the manifest's
    header_pattern), with any of the four as named groups, for a header not
    in the Santhya's shape; without one the Santhya's rules apply.
    """
    out = {"ang_from": None, "ang_to": None, "book_page": None, "section": None}
    if not text:
        return out
    if pattern is not None:
        pat = re.compile(pattern) if isinstance(pattern, str) else pattern
        m = pat.search(text)
        if not m:
            return out
        g = m.groupdict()
        out["ang_from"] = to_int(g["ang_from"]) if g.get("ang_from") else None
        out["ang_to"] = to_int(g["ang_to"]) if g.get("ang_to") else out["ang_from"]
        if out["ang_from"] and out["ang_to"] and out["ang_to"] < out["ang_from"]:
            out["ang_to"] = out["ang_from"]
        out["book_page"] = to_int(g["book_page"]) if g.get("book_page") else None
        out["section"] = g["section"].strip() if g.get("section") else None
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
    # a watermark laid across the page ("SIKHBOOKCLUB.COM" on the diagonal) has a box as tall as the
    # middle of the page: the lines under it are the book's own, and only its words are the stamp's
    # (is_stamp). A stamp that is a line of text -- "Page 41 of 530" -- claims what lies in its box.
    line_stamps = [s for s in (stamps or []) if (s["bbox"][3] - s["bbox"][1]) <= max(3 * body_h, 0.05 * page_h)]
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
        elif is_stamp(text) or any(_overlaps(rec["bbox"], s["bbox"]) for s in line_stamps):
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


def vertical_rule(img, page_w: int | None = None) -> dict | None:
    """
    {"x", "y0", "y1"}: the centre column of a hairline rule down the page
    body with white either side, and the rows it spans; None where the page
    has none.

    A rule is a narrow run of dark pixel columns in which one unbroken
    vertical stroke spans a good part of the page (RULE_MIN_RUN): a column
    of text inks as much of the page as a short rule does, but never in one
    stroke longer than a line. Of several candidates the longest stroke wins
    -- a table's inner line or a picture's edge is shorter than a rule drawn
    beside a whole band of verse. The rows matter as much as the column: in
    the Santhya the rule runs beside the paired verse-and-arth band only, and
    the prose above and below it spans the whole page (p.94, 216, 344).
    """
    import numpy as np
    h, w = img.shape[:2]
    pw = page_w or w
    ink = img < 128
    lo, hi = int(pw * 0.2), min(int(pw * 0.8), w)
    gap, max_w, min_run = max(2, int(pw * RULE_GAP)), max(1, int(pw * RULE_MAX_W)), int(h * RULE_MIN_RUN)
    band = (int(h * RULE_BAND[0]), int(h * RULE_BAND[1]))

    # The longest vertical stroke in every pixel column at once, a gap of up
    # to three rows bridged (a rule's halo thins where the scan dithered it).
    # Per column rather than over a run of dark columns: a rule set tight
    # against a line of text shares that line's rows, and judged together
    # they are one wide dark band.
    n_cols = ink.shape[1]
    run = np.zeros(n_cols, dtype=np.int32)
    gaps = np.zeros(n_cols, dtype=np.int32)
    start = np.zeros(n_cols, dtype=np.int32)
    best_len = np.zeros(n_cols, dtype=np.int32)
    best_r0 = np.zeros(n_cols, dtype=np.int32)
    best_r1 = np.zeros(n_cols, dtype=np.int32)
    for i, row in enumerate(ink):
        fresh = row & (run == 0)
        start = np.where(fresh, i, start)
        run = np.where(row, np.where(fresh, 1, run + gaps + 1), run)
        gaps = np.where(row, 0, gaps + 1)
        run = np.where(gaps > 3, 0, run)
        better = run > best_len
        best_len = np.where(better, run, best_len)
        best_r0 = np.where(better, start, best_r0)
        best_r1 = np.where(better, i + 1, best_r1)

    def white_beside(profile, at: int, step: int) -> int:
        # the white touching the rule, its halo (a few grey columns where the
        # scan dithered the stroke) stepped over first
        x = at
        skipped = 0
        while 0 <= x < w and profile[x] >= RULE_GAP_INK and skipped < 4:
            skipped += 1
            x += step
        n = 0
        while 0 <= x < w and profile[x] < RULE_GAP_INK:
            n += 1
            x += step
        return n

    best: tuple[int, int, int, int] | None = None
    x = lo
    while x < hi:
        if best_len[x] < min_run:
            x += 1
            continue
        end = x + 1
        while end < hi and best_len[end] >= min_run:
            end += 1
        width = end - x
        if width <= max_w:
            c = x + width // 2
            # the rows: the union over the rule's columns, since the scan
            # breaks the stroke in one column and not the next
            stroke = int(best_len[x:end].max())
            r0, r1 = int(best_r0[x:end].min()), int(best_r1[x:end].max())
            # white touching the rule on both sides (a halo of a pixel or two
            # excused), judged over the rule's own rows: beside a rule that
            # runs down half the page, the other half's text says nothing
            # about the gutter
            beside = ink[r0:r1].mean(axis=0)
            if (r1 > band[0] and r0 < band[1]
                    and white_beside(beside, x - 1, -1) >= gap and white_beside(beside, end, 1) >= gap):
                if best is None or stroke > best[0]:
                    best = (stroke, c, r0, r1)
        x = end
    return {"x": best[1], "y0": best[2], "y1": best[3]} if best else None


def find_vertical_rule(img, page_w: int | None = None) -> int | None:
    """The centre column of the page's hairline rule, or None (vertical_rule without the rows)."""
    rule = vertical_rule(img, page_w)
    return rule["x"] if rule else None


def page_columns_with_source(words: list[dict], page_w: int, img=None, lines: list[dict] | None = None,
                             rule_x: int | None = None) -> tuple[list[tuple[int, int]], str | None]:
    """
    ([(x_lo, x_hi), ...], how) for a two-column page, ([], None) for one
    column; `how` is "rule" (a hairline the printer drew, find_vertical_rule),
    "ink" (an empty band read off the image, columns_by_ink) or "words" (a
    band no word begins in, as lib/writings_pdf.columns does).
    """
    if rule_x is not None:
        xs = [w["bbox"][0] for w in words if _LETTERS.search(w.get("text", ""))]
        left = sum(1 for x in xs if x < rule_x)
        # a rule with words on both sides of it is a gutter, whatever the
        # engine's line boxes say: lines across it are what split_at_gutter cuts
        if len(xs) >= 10 and min(left, len(xs) - left) >= COLUMN_SHARE_RULE * len(xs):
            return [(0, rule_x), (rule_x, page_w)], "rule"
    if img is not None:
        found = columns_by_ink(img, words, page_w, lines)
        if found:
            return found, "ink"
        # the word-start profile may still find it (a rule read as ink on a
        # dark scan); the line-crossing test decides either way
        found = page_columns(words, page_w)
        if found and lines:
            at = found[0][1]
            body_lines = [ln for ln in lines if ln.get("text", "").strip() and ln.get("zone") in (None, "body")]
            crossing = sum(1 for ln in body_lines if ln["bbox"][0] < at - 20 and ln["bbox"][2] > at + 20)
            if body_lines and crossing > CROSSING_MAX * len(body_lines):
                return [], None
        return found, ("words" if found else None)
    found = page_columns(words, page_w)
    return found, ("words" if found else None)


def page_columns(words: list[dict], page_w: int, img=None, lines: list[dict] | None = None,
                 rule_x: int | None = None) -> list[tuple[int, int]]:
    """
    [(x_lo, x_hi), ...] for a two-column page, [] for one column.
    With the page image the gutter is read from the ink (columns_by_ink);
    without it, from where words BEGIN, as lib/writings_pdf.columns does;
    a hairline rule (rule_x) is believed before either.
    """
    if img is not None or rule_x is not None:
        return page_columns_with_source(words, page_w, img, lines, rule_x)[0]
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


def header_of(lines: list[dict], pattern=None) -> dict:
    """Hints from every header-zone line on a page, first non-empty value wins."""
    out = {"ang_from": None, "ang_to": None, "book_page": None, "section": None}
    for ln in lines:
        if ln.get("zone") != "header":
            continue
        h = header_hints(ln.get("text", ""), pattern)
        for k, v in h.items():
            if out[k] is None and v is not None:
                out[k] = v
    return out


HINT_WINDOW = 3            # known pages either side a page's ang is judged against
HINT_SLACK = 3             # angs a page may differ from what its neighbours predict, at least


def smooth_hints(by_page: dict, window: int = HINT_WINDOW, slack: int = HINT_SLACK) -> dict:
    """
    The pages' header hints with the misread angs put right.

    A running header is read by the same OCR as the body, and a digit
    misread turns ang ੧੨ into ੯੨ on one page in five of the Santhya (84 of
    437 headers): the ang is the window the corpus match looks in, so on
    those pages the match was looking 80 angs away. Angs advance steadily
    through a book, so each page's ang is judged against what its nearest
    known neighbours predict at the book's own rate (angs per page): a value
    off by more than `slack` (or three times what the rate says the window
    spans) is replaced by the interpolation between the trusted neighbours,
    and a page with no header takes the same. Each page says how it came by
    its ang: "header" (as read), "smoothed" (put right or filled in), or
    None (nothing to go on). book_page is fixed the same way, by the modal
    offset between it and the scan's page number. Running it twice changes
    nothing.

    @param by_page  {page: hints} as header_of() returns them
    @returns        {page: hints} with ang_from, ang_to, book_page, ang_source set
    """
    import statistics
    pages = sorted(by_page)
    known = [(p, by_page[p]["ang_from"]) for p in pages if by_page[p].get("ang_from")]
    out = {p: dict(by_page[p]) for p in pages}
    if not known:
        for p in pages:
            out[p]["ang_source"] = None
        return out
    # the book's rate: angs per page over consecutive known pages that advance
    steps = [(a2 - a1) / float(p2 - p1) for (p1, a1), (p2, a2) in zip(known, known[1:]) if 0 <= a2 - a1 <= 10 and p2 > p1]
    rate = statistics.median(steps) if steps else 0.0
    tolerance = max(slack, int(round(3 * rate * window)))
    trusted: dict = {}
    for i, (p, a) in enumerate(known):
        neighbours = known[max(0, i - window):i] + known[i + 1:i + 1 + window]
        if not neighbours:
            trusted[p] = a
            continue
        predicted = statistics.median(aq + rate * (p - q) for q, aq in neighbours)
        if abs(a - predicted) <= tolerance:
            trusted[p] = a
    keys = sorted(trusted)
    for p in pages:
        h = out[p]
        if p in trusted:
            # a value this pass agrees with stays what it was: read from the
            # header, or filled in by an earlier pass
            h["ang_source"] = "smoothed" if h.get("ang_source") == "smoothed" else "header"
            if not h.get("ang_to") or h["ang_to"] < h["ang_from"] or h["ang_to"] - h["ang_from"] > tolerance:
                h["ang_to"] = h["ang_from"]
            continue
        before = [q for q in keys if q < p]
        after = [q for q in keys if q > p]
        if before and after:
            a, b = before[-1], after[0]
            value = int(trusted[a] + (trusted[b] - trusted[a]) * (p - a) / float(b - a))
        elif before or after:
            q = before[-1] if before else after[0]
            value = int(round(trusted[q] + rate * (p - q)))
        else:
            value = None
        h["ang_from"] = h["ang_to"] = value
        h["ang_source"] = "smoothed" if value is not None else None
    # the book's own page number runs one ahead per scan page: its offset from
    # the scan's numbering is one number for the whole book, and a misread
    # page number is the one that breaks it
    offsets = [p - out[p]["book_page"] for p in pages if out[p].get("book_page")]
    if offsets:
        mode = statistics.mode(offsets)
        for p in pages:
            bp = out[p].get("book_page")
            if bp is None or abs((p - bp) - mode) > 2:
                out[p]["book_page"] = p - mode
    return out
