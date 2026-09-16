"""
Reading an author's essays out of PDF, paragraph by paragraph.

The Bau Ji essays come as 126 PDFs. Every one carries a real text layer
(embedded TrueType with ToUnicode), so nothing here does OCR. Two shapes occur:

  spread   the Punjabi/ folder: one PDF page is a two-up spread, Punjabi left and
           English right. The Punjabi half is a SCANNED IMAGE -- it contributes
           no text beyond its page marker -- so the text layer of a spread is
           already the English half. The shape is detected from where lines
           start, and the stray left-hand marker is dropped.
  single   the English/ folder and the five books: one ordinary column.

Two signals are lifted from page geometry rather than guessed from wording:

  italic   Gurbani quotations are set in italic, and the italic is SYNTHETIC --
           a shear in the text matrix (measured tm[2]/tm[0] ~ 0.34 on quotes,
           0.0 on body), not a separate font. On the spreads the sheared runs are
           exactly the quoted translations, which makes this the quote detector
           for those files. The single-column files carry no shear and fall back
           to the `|| n ||` and ang markers lib/citations.py looks for.
  indent   a paragraph's first line is indented past the page's left margin;
           continuation lines sit on it. Breaking on "indented" rather than on
           "x differs" is what keeps a paragraph whole.

Run joining is calibrated, not guessed: across 261 same-line run pairs the
advance of a contiguous run is 0.38-0.46 of (characters x size) and a real word
gap is 0.64 or more. A run's end is therefore estimated at CHAR_W per character
and a space inserted only when the next run starts SPACE_W of a size beyond it.

Page markers ("L127.1", "L68/3") are the essay number and the page within it,
captured before the running-header filter discards them because they are the
only in-document identifier an essay has.

Nothing here interprets meaning; it returns paragraphs with enough provenance
that re-chunking never requires re-reading the PDF.
"""
from __future__ import annotations
import re
import statistics

from pypdf import PdfReader

# "L127.1", "L68/3" -- essay number and page within the essay. A spread carries
# the left half's marker and the right half's on the same line, so we search
# rather than anchor, and keep the last (the English half).
MARKER = re.compile(r"L\s*([0-9]{1,4})\s*[./]\s*([0-9]{1,3})\b", re.I)
PAGE_NO = re.compile(r"^[0-9]{1,3}$")       # a line that is only a small number
ITALIC_RATIO = 0.10                          # tm[2]/tm[0]; 0.34 on quotes, 0.0 on body
CHAR_W = 0.46                                # mean advance per character (calibrated, p75)
SPACE_W = 0.25                               # a gap wider than this fraction of the size is a space
SPREAD_X = 380.0                             # the English column of a spread starts near 412
GUTTER = 80.0                                # an empty x band this wide separates two columns
COLUMN_SHARE = 0.25                          # each column must hold at least this share of the runs
BIN = 4.0                                    # x resolution of the start-position profile
GUTTER_TOL = 0.005                           # share of runs allowed to begin inside a gutter
Y_TOL = 2.5                                  # runs within this many units share a line
INDENT = 6.0                                 # a first line sits this far past the margin
# A paragraph breaks on a gap wider than this many times the page's own median
# line gap. Measured against the font size, gaps are bimodal (1.25 within a
# paragraph, 2.25 or more between), but a fixed multiple of the size is WORSE
# here than the page median, and both alternatives were tried:
#   per-line size x 2.0   quotes are set smaller than body, so the threshold
#                         shrinks inside a quote block and cuts the trailing ang
#                         off its verse -- quotes carrying an ang fell 71.5% -> 48.6%
#   body size x 2.0       same failure, 45.3%
# The page median adapts to what is actually on the page, which matters because
# half this author's paragraphs are one-word items in a cascading list, and such
# a page's gaps are genuinely wider throughout.
#
# It leaves 4.4% of body paragraphs ending mid-sentence, from first-line indents
# inside justified text rather than from leading (swept 1.8-3.0, barely moves).
# That is the recoverable direction: 12_ingest_writings.py rejoins a paragraph
# that ends mid-sentence with the one continuing it in lower case, which
# under-segmenting could never undo.
LEADING_BREAK = 1.6


def page_runs(page) -> list[dict]:
    """Every non-blank text run on the page with its position, size and slant."""
    out: list[dict] = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        scale = abs(tm[0]) or 1.0
        out.append({"x": tm[4], "y": tm[5], "size": scale,
                    "italic": abs(tm[2]) / scale > ITALIC_RATIO,
                    "text": text.replace("\n", " ")})

    page.extract_text(visitor_text=visit)
    return out


def join_runs(runs: list[dict]) -> str:
    """Concatenate runs left to right, inserting a space only at a real gap."""
    parts: list[str] = []
    prev = None
    for r in runs:
        if prev is not None and not r["text"].startswith(" ") and not parts[-1].endswith(" "):
            end = prev["x"] + len(prev["text"]) * prev["size"] * CHAR_W
            if r["x"] - end > SPACE_W * prev["size"]:
                parts.append(" ")
        parts.append(r["text"])
        prev = r
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def group_lines(runs: list[dict]) -> list[dict]:
    """Runs merged into lines by y position, each line read left to right."""
    lines: list[dict] = []
    for r in sorted(runs, key=lambda r: (-r["y"], r["x"])):
        if lines and abs(lines[-1]["y"] - r["y"]) <= Y_TOL:
            lines[-1]["runs"].append(r)
        else:
            lines.append({"y": r["y"], "runs": [r]})
    for ln in lines:
        rs = sorted(ln["runs"], key=lambda r: r["x"])
        ln["x0"] = rs[0]["x"]
        ln["size"] = statistics.median([r["size"] for r in rs])
        ink = sum(len(r["text"]) for r in rs)
        ln["italic"] = sum(len(r["text"]) for r in rs if r["italic"]) > ink / 2
        ln["text"] = join_runs(rs)
    return lines


def columns(pages: list[list[dict]]) -> list[tuple[float, float]]:
    """
    The document's text columns, left to right, or [] for a single column.

    The English/ essays set prose in a left column and more prose in a right one,
    with a gutter between. Grouping such a page by y alone interleaves the two
    ("...their natural evolution (towards the In doing so man is cheating
    himself..."), which destroys the sense of both, so the columns are found
    first and read one after the other.

    The gutter is found from where runs BEGIN, over the whole document. Two
    other things were tried and do not work: the widest gap on a single page
    (one stray run bridging the gutter moves it to a wide gap inside a column --
    measured on page 3 of Hukam Part 1), and an ink-coverage profile (a run's
    width has to be estimated, and the estimate overshoots enough to paint the
    gutter solid). Where a line starts is exact, needs no estimate, and holds
    across a document even when one page is laid out differently.

    Measured: the two-column essays show a gutter 104-208 units wide with about
    half the runs on each side; the books show 24-28 and are single column.

    A two-up spread is not two columns by this test -- its left half is a scanned
    image contributing one page marker, far below COLUMN_SHARE.
    """
    runs = [r for page in pages for r in page]
    xs = [r["x"] for r in runs]
    if len(xs) < 40:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 3 * GUTTER:
        return []
    bins = int((hi - lo) / BIN) + 1
    starts = [0] * bins
    for x in xs:
        starts[int((x - lo) / BIN)] += 1
    # A few strays must not close the gutter: one run beginning in it on one
    # page of sixty is enough to hide it if the test is "nothing at all".
    limit = max(1, int(GUTTER_TOL * len(xs)))
    best, run_from = (0, None), None
    for i, v in enumerate(starts):
        quiet = v <= limit and 0.15 < i / bins < 0.85
        if quiet and run_from is None:
            run_from = i
        elif not quiet and run_from is not None:
            if i - run_from > best[0]:
                best = (i - run_from, (run_from + i) / 2.0)
            run_from = None
    if best[0] * BIN < GUTTER or best[1] is None:
        return []
    at = lo + best[1] * BIN
    left = sum(1 for x in xs if x < at)
    if min(left, len(xs) - left) < COLUMN_SHARE * len(xs):
        return []
    return [(lo - 1.0, at), (at, max(r["x"] for r in runs) + 1e6)]


def is_spread(lines: list[dict]) -> bool:
    """True when the page is two-up and only the right (English) half has text."""
    starts = [ln["x0"] for ln in lines if len(ln["text"]) > 12]
    if len(starts) < 4:
        return False
    return sum(1 for x in starts if x >= SPREAD_X) >= 0.8 * len(starts)


def left_margin(lines: list[dict]) -> float:
    """The x most lines start at -- the margin a first-line indent is measured from."""
    starts = [round(ln["x0"] / 2.0) * 2.0 for ln in lines if len(ln["text"]) > 20]
    if not starts:
        return 0.0
    return statistics.mode(starts) if len(set(starts)) < len(starts) else min(starts)


def paragraphs(lines: list[dict], body_size: float, margin: float) -> list[dict]:
    """
    Lines gathered into paragraphs.

    A paragraph breaks when the italic flag flips (a quoted verse against the
    prose around it), when the size changes, on a wider-than-usual vertical gap,
    or -- for body text only -- when a line is indented past the margin, which is
    how a new paragraph announces itself. Quoted blocks deliberately do NOT break
    on indent: a verse's own lines are indented relative to its leading number,
    and splitting there would cut a couplet in half.
    """
    lines = [ln for ln in lines if ln["text"]]
    if not lines:
        return []
    gaps = [lines[i - 1]["y"] - lines[i]["y"] for i in range(1, len(lines))]
    normal = statistics.median(gaps) if gaps else 0.0
    out: list[dict] = []
    cur = None
    for i, ln in enumerate(lines):
        style = "quote" if ln["italic"] else (
            "heading" if ln["size"] >= body_size * 1.15 and len(ln["text"]) < 80 else "body")
        gap = (lines[i - 1]["y"] - ln["y"]) if i else 0.0
        indented = style == "body" and ln["x0"] > margin + INDENT
        if (cur is None or style != cur["style"] or abs(ln["size"] - cur["size"]) > 1.0
                or (normal and gap > normal * LEADING_BREAK) or indented):
            cur = {"x0": ln["x0"], "size": ln["size"], "italic": ln["italic"],
                   "style": style, "text": ln["text"]}
            out.append(cur)
        elif cur["text"].endswith("-"):
            # these files break lines between words, so a trailing hyphen is a
            # real compound ("self-willed") and is kept, joined without a space
            cur["text"] += ln["text"]
        else:
            cur["text"] += " " + ln["text"]
    for p in out:
        p["text"] = re.sub(r"\s+", " ", p["text"]).strip()
    return out


def read_pdf(path: str) -> dict:
    """
    One PDF as pages of paragraphs.

    @returns {"path", "body_size", "pages": [{"page", "marker", "spread", "paragraphs"}]}
    where a paragraph is {"text", "style", "italic", "x0", "size"}.
    """
    reader = PdfReader(path)
    per_page: list[dict] = []
    for n, page in enumerate(reader.pages, start=1):
        try:
            per_page.append({"page": n, "runs": page_runs(page)})
        except Exception as exc:              # a damaged page must not lose the book
            per_page.append({"page": n, "runs": [], "error": str(exc)})
    bands = columns([p["runs"] for p in per_page])

    sizes: list[float] = []
    raw: list[dict] = []
    for p in per_page:
        runs = p["runs"]
        cols = ([group_lines([r for r in runs if lo <= r["x"] < hi]) for lo, hi in bands]
                if bands else [group_lines(runs)])
        every = [ln for col in cols for ln in col]
        raw.append({"page": p["page"], "cols": cols, "spread": is_spread(every),
                    "columns": len(bands), **({"error": p["error"]} if p.get("error") else {})})
        sizes.extend(ln["size"] for ln in every if len(ln["text"]) > 20)
    body = statistics.median(sizes) if sizes else 10.0

    pages = []
    for pg in raw:
        marker, out = None, []
        for col in pg.get("cols", []):
            kept = []
            for ln in col:
                text = ln["text"].strip()
                found = MARKER.findall(text.replace(" ", ""))
                if found and len(text) < 32:
                    essay, page_no = found[-1]   # the English half's marker wins
                    marker = marker or f"L{int(essay)}.{int(page_no)}"
                    continue
                if PAGE_NO.match(text):
                    continue
                if pg.get("spread") and ln["x0"] < SPREAD_X:
                    continue                     # stray ink from the scanned Punjabi half
                kept.append(ln)
            out.extend(paragraphs(kept, body, left_margin(kept)))
        pages.append({
            "page": pg["page"], "marker": marker, "spread": bool(pg.get("spread")),
            "columns": pg.get("columns", 0), "paragraphs": out,
            **({"error": pg["error"]} if pg.get("error") else {}),
        })
    return {"path": path, "body_size": body, "pages": pages}
