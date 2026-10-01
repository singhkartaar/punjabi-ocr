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

Run joining is measured where it can be and calibrated where it cannot. A PDF
states the advance of every glyph it embeds, and where those widths are present
(`font_metrics`) the end of a run is known exactly and the gap that remains IS
the space. Where they are absent -- Bau Ji's scans give none -- the older
calibration stands in: across 261 same-line run pairs the
advance of a contiguous run is 0.38-0.46 of (characters x size) and a real word
gap is 0.64 or more. A run's end is therefore estimated at CHAR_W per character
and a space inserted only when the next run starts SPACE_W of a size beyond it.

Furniture -- running heads, printed page numbers, watermarks -- is stripped only
when `read_pdf(furniture=True)` asks. See that function for why it is opt-in.

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
EXACT_SPACE = 0.12                            # ... but half that when the run end is exact (see join_runs)
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


ITALIC_NAME = re.compile(r"italic|oblique", re.I)


def font_metrics(font_dict) -> tuple | None:
    """
    `(first_char, widths, space_width)` from an embedded font, or None.

    A PDF states the advance of every glyph it embeds, in thousandths of the em.
    Reading them removes the guesswork below: where they are present the end of
    a run is known rather than estimated, and the width of the font's OWN space
    is what decides whether a gap is one.
    """
    try:
        first, widths = font_dict.get("/FirstChar"), font_dict.get("/Widths")
        if first is None or not widths:
            return None
        first = int(first)
        widths = [float(w) for w in widths]
        i = 32 - first                                   # the space glyph
        space = widths[i] if 0 <= i < len(widths) and widths[i] else None
        return first, widths, space
    except Exception:
        return None                                      # a font we cannot read is a fallback


def run_width(text: str, metrics, size: float):
    """The exact advance of `text`, or None if any glyph is unknown."""
    if not metrics:
        return None
    first, widths, _ = metrics
    total = 0.0
    for ch in text:
        i = ord(ch) - first
        if not (0 <= i < len(widths)) or not widths[i]:
            return None
        total += widths[i]
    return total / 1000.0 * size


# A conventionally typeset book sets ff, fi and fl as SINGLE glyphs, so
# `suffering` arrives as "suﬀering" -- one character no vocabulary knows,
# no tokeniser splits, and a reader sees on the page. Bandginama has ~700.
# Expand these six and nothing else: blanket NFKC would also rewrite fractions,
# superscripts and some Gurmukhi composition, none of which is broken here.
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
             "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
LIGATURE = re.compile("[ﬀ-ﬆ]")


def unligature(text: str) -> str:
    """Expanded AFTER the run is measured: a ligature is one glyph, not two, and
    the PDF states ITS width -- measuring "ff" instead would widen the run and
    swallow the space after it."""
    return LIGATURE.sub(lambda m: LIGATURES[m.group()], text)


def page_runs(page) -> list[dict]:
    """Every non-blank text run on the page with its position, size and slant."""
    out: list[dict] = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        scale = abs(tm[0]) or 1.0
        m = font_metrics(font_dict) if font_dict else None
        name = str(font_dict.get("/BaseFont")) if font_dict else ""
        # Measure the INK, not the string pypdf hands over. pypdf adds a space
        # or a newline of its own wherever it reads a displacement as one, and
        # a Word-set book (Calibri, whose subset has no space glyph) gets both
        # wrong: "Thos\n" + "e" is one word split by kerning -- 1,301 of the
        # 1,307 such breaks in In Search of the True Guru touch the next run --
        # and " their" measured with its space meets no glyph, falls back to
        # the estimate, and the estimate overshoots and swallows the real gap
        # ("theirhearts"). So the glyphs are measured on their own, and the
        # newline is dropped where the ink is exact and the gap can decide. It
        # stays a space where it cannot, which is every scan without metrics.
        # And a font whose subset has no space glyph never SET a space, so a
        # space inside one of its runs is pypdf's reading of a kerning
        # displacement -- "an d" + "said," measured 2.7 units short of the
        # next run exactly where "and" would be -- and is dropped for the
        # same reason: the gap to the next run decides.
        if m and not m[2]:
            text = text.replace(" ", "")
        ink = text.strip()
        width = run_width(ink, m, scale)
        text = text.replace("\n", "" if width is not None else " ")
        out.append({"x": tm[4], "y": tm[5], "size": scale,
                    # Two ways to be italic, and a book uses one or the other:
                    # a synthetic slant in the text matrix, or a real italic
                    # face. Bau Ji's scans are the first, typeset books the
                    # second -- testing only for the slant makes every quote in
                    # a typeset book read as body.
                    "italic": abs(tm[2]) / scale > ITALIC_RATIO or bool(ITALIC_NAME.search(name)),
                    "text": unligature(text),
                    "width": width,
                    "space": (m[2] / 1000.0 * scale) if m and m[2] else None})

    page.extract_text(visitor_text=visit)
    return out


def join_runs(runs: list[dict]) -> str:
    """
    Concatenate runs left to right, inserting a space only at a real gap.

    These PDFs emit one run per word with no separator of their own, so every
    space in the output is inferred here -- which makes getting it wrong the
    difference between prose and `On hearingthis,theInspectorGeneralwas`.
    Where the font states its metrics both sides of the test are exact: the end
    of the previous run, and half the width of that font's own space. Where it
    does not, the calibrated estimate above is used instead.
    """
    parts: list[str] = []
    prev = None
    for r in runs:
        if prev is not None and not r["text"].startswith(" ") and not parts[-1].endswith(" "):
            if prev.get("width") is not None:
                # With an exact end, the gap that remains IS the space, so the
                # test is much tighter than the estimated case: a word boundary
                # measures ~0.22 em and a run split mid-word ~0. Many subset
                # fonts drop the space glyph entirely (FirstChar 33) because the
                # PDF positions every word itself, so its own width is often
                # unavailable and EXACT_SPACE stands in.
                end = prev["x"] + prev["width"]
                gap = prev["space"] * 0.5 if prev.get("space") else EXACT_SPACE * prev["size"]
            else:
                end = prev["x"] + len(prev["text"]) * prev["size"] * CHAR_W
                gap = SPACE_W * prev["size"]
            if r["x"] - end > gap:
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


# A magazine page, not a book: two columns on one page, four on the next, one
# on a third; a pull quote or an indented verse starting inside a gutter; and
# too few lines in a three-page article for the whole-document profile above to
# outvote them (Eternal Voice, rosters/barusahib.json: columns() finds nothing
# in any of its sixteen articles). So each page is asked on its own where its
# columns START: an x at which at least PAGE_COL_SHARE of the page's lines (and
# PAGE_COL_MIN of them) begin, at least PAGE_COL_GAP of the width from the
# column before. A run starting mid-line in a single-column page -- an italic
# word, a bold name -- is a scattering of x's, never a share of the page at one,
# and an indented verse sits well inside PAGE_COL_GAP of its margin.
PAGE_COL_SHARE = 0.10
PAGE_COL_MIN = 6
PAGE_COL_GAP = 0.20
PAGE_COL_BIN = 12.0                          # x tolerance of "one x"


# Fake bold: a heading drawn several times over. A PDF's text layer keeps the
# copies back to back -- the whole run ("Union with the Divine (God)Union with
# the Divine (God)..."), or a label inside a run of body text ("Inert Matter :
# Inert Matter : ... This class consists of suns") -- or as separate runs a
# fraction of a point apart ("Guru Guru Guru Guru Guru Arjan Dev states:"). The
# book's own count of copies is the roster's (`overprint`: 5 for Sikh Faith,
# rosters/barusahib.json), and a stretch of three or more characters repeated
# exactly that many times, back to back, is cut to one copy: prose repeats a
# word, never a string five times with nothing between.
OVERPRINT_NEAR = 3.0                         # a copy drawn this close is the same run
CAPITAL_APART = re.compile(r"\b([A-Z]) (?=[a-z]{3,})")


def collapse_overprint(runs: list[dict], copies: int = 5) -> int:
    """
    Cut each overprinted stretch inside a run to one copy, in place; returns
    how many runs were cut. Copies drawn as separate runs are drop_shadows'.

    Three passes, the later two only on a run the first has already shown to be
    fake bold:
      1. a stretch of three or more characters repeated exactly `copies` times
         back to back -- except one word and a space ("Har Har Har Har Har" is
         how a verse or a chant is written);
      2. in a fake-bold run, that one word too: "Guru Guru Guru Guru Guru Arjan
         Dev states:" held "Arjan Dev states:" five times as well;
      3. in a fake-bold run, a word's first letter stuttered before it -- "T T
         T T Temptation", "W WW WWilliam" -- where the copies of the letter did
         not land on the word, and a capital left standing apart from its word
         ("William W arburt").
    """
    unit = re.compile(r"(.{3,}?)\1{%d}(?!\1)" % (copies - 1))
    stutter = re.compile(r"\b(\w)(?: ?\1){1,%d} ?(?=\1\w)" % (copies - 1))
    cut = 0
    for r in runs:
        text = unit.sub(lambda m: m.group(0) if re.fullmatch(r"\S+ ", m.group(1)) else m.group(1),
                        r["text"])
        if text == r["text"]:
            continue
        text = unit.sub(r"\1", text)
        text = stutter.sub("", text)
        # the capital a stutter split off its word: "William W arburt"
        text = CAPITAL_APART.sub(r"\1", text)
        if r.get("width"):
            r["width"] = r["width"] * len(text) / max(1, len(r["text"]))
        r["text"] = text
        cut += 1
    return cut


def drop_shadows(runs: list[dict]) -> int:
    """
    Drop a run whose text an earlier run already has, drawn within
    OVERPRINT_NEAR in both x and y; returns how many were dropped.

    Fake bold as a drop shadow: the Eternal Voice articles set a heading twice,
    a point and a half apart ("Need for" at 56.00, 610.55 and at 54.67,
    612.26), and read by lines the two interleave -- "Need for Need for
    God-Conscious God-Conscious PersonsPersons". Text does not repeat itself on
    the same spot, so this is safe for any file that turns it on.
    """
    kept: list[dict] = []
    dropped = 0
    for r in runs:
        text = r["text"].strip()
        if text and any(k["text"].strip() == text and abs(k["x"] - r["x"]) < OVERPRINT_NEAR
                        and abs(k["y"] - r["y"]) < OVERPRINT_NEAR for k in kept[-12:]):
            dropped += 1
            continue
        kept.append(r)
    runs[:] = kept
    return dropped


def legacy_font_run(text: str) -> bool:
    """
    A run of Gurmukhi set in a legacy font this pipeline cannot convert.

    Sant Waryam Singh Ji's books (rosters/ratwara.json) print every verse three
    ways -- the English rendering with its ang, the Gurmukhi, and the
    explanation -- and the Gurmukhi is in two fonts that are not GurbaniAkhar,
    the one anvaad converts: Amar Gatha's maps the letters onto Latin-1
    ("\u00d5\u00c7\u00f0 \u00d5\u00c7\u00f0 \u00d4\u00c5\u00c7\u00f0\u00fa", ਕਰਿ ਕਰਿ ਹਾਰਿਓ),
    Surat Shabad Marg's onto ASCII punctuation and consonants
    ("frqj ;'fJB uzdB[", ਗ੍ਰਿਹ ਸੋਇਨ ਚੰਦਨ). Read as English either is noise in
    the passage, and the verse is not lost: its English carries the ang, and
    the citation made from that shows the reader the Gurmukhi from the corpus.

    Latin-1: at least a third of the characters in U+00A0-U+00FF (an English
    line has an accent or none). ASCII: eight letters or more and under a
    fifth of them vowels -- 8% on the line above, 35-40% on English, 30% on
    romanised Gurbani.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    if sum(1 for c in chars if "\u00a0" <= c <= "\u00ff") >= len(chars) / 3:
        return True
    letters = [c for c in chars if c.isascii() and c.isalpha()]
    if len(letters) < 8 or sum(1 for c in letters if c in "aeiouAEIOU") >= 0.2 * len(letters):
        return False
    # few vowels alone is not enough -- "strength and rhythm" has two in
    # sixteen -- so the font must also show itself: its punctuation letters
    # (";" is sa, "[" is aunkar) or a capital inside a word ("fJB", "rzX")
    return bool(re.search(r"[;\[\]{}/]|[a-z][A-Z]", text))


# Words of the ASCII legacy font inside a line of English -- Surat Shabad Marg
# sets the Gurmukhi and the English in one run: "... ;woE.. v'bB s/ okyj[ gqG{
# BkBe d/ efo jE.. P. 256 'After wandering and wandering O Lord ...". The font
# shows itself where English never puts the same marks: ";" (sa) before a
# letter, "[" "{" (aunkar, dulainkar) after one, "/" (lavan) ending a word or
# after a single letter, "?" after a single letter, a capital inside a word,
# the ".." danda, and a word with no vowel at all ("Bkw", "Jhx").
LEGACY_MARKS = re.compile(r";[A-Za-z'\[{]|[A-Za-z;'][\[{]|[A-Za-z]/(?:[.,'\"]*$)"
                          r"|(?<![A-Za-z])[A-Za-z]/[A-Za-z]|(?<![A-Za-z])[A-Za-z]\?$|[a-z][A-Z]|\.\.$")
# words English uses short and often: never a bridge between two legacy words
ENGLISH_SHORT = set("the and of to is in that a are be for with as by this it on or his he you your we our "
                    "not but from have has was were will shall all one who which when then there their them "
                    "they so if no at an my me i o lord god guru name mind soul thou thy thee".split())


def legacy_word(t: str) -> bool:
    if LEGACY_MARKS.search(t):
        return True
    letters = re.sub(r"[^A-Za-z]", "", t)
    return len(letters) >= 2 and not re.search(r"[aeiouyAEIOUY]", letters)


def strip_legacy_words(text: str) -> str:
    """
    A paragraph without its stretches of legacy-font Gurmukhi. A stretch is two
    or more legacy words, joined across a short word (seven letters or fewer,
    not a common English one) when another legacy word follows within three.
    Measured on the three Ratwara Sahib books: 516 stretches, none English.
    """
    toks = text.split(" ")
    marks = [bool(t) and legacy_word(t) for t in toks]
    n = len(toks)
    drop = [False] * n
    i = 0
    while i < n:
        if not marks[i]:
            i += 1
            continue
        j, count, k = i, 1, i + 1
        while k < n:
            if marks[k]:
                count, j, k = count + 1, k, k + 1
                continue
            bare = re.sub(r"[^A-Za-z]", "", toks[k])
            if toks[k] == "" or (len(bare) <= 7 and bare.lower() not in ENGLISH_SHORT
                                 and any(marks[m] for m in range(k + 1, min(n, k + 4)))):
                k += 1
                continue
            break
        if count >= 2:
            for m in range(i, j + 1):
                drop[m] = True
        i = j + 1
    return re.sub(r"\s{2,}", " ", " ".join(t for t, d in zip(toks, drop) if not d)).strip()


def page_columns(runs: list[dict], width: float) -> list[tuple[float, float]]:
    """This page's columns, left to right, split where each one starts, or [] for one."""
    xs = [r["x"] for r in runs if r["text"].strip()]
    if not xs:
        return []
    bins: dict = {}
    for x in xs:
        bins.setdefault(int(x // PAGE_COL_BIN), []).append(x)
    # a start with its neighbouring bins, so a bin edge cannot halve it
    starts = []
    for k in sorted(bins):
        near = bins[k] + bins.get(k - 1, []) + bins.get(k + 1, [])
        if len(near) >= max(PAGE_COL_MIN, PAGE_COL_SHARE * len(xs)):
            starts.append((min(near), len(near)))
    cols: list[float] = []
    for x, n in starts:
        if not cols or x - cols[-1] >= PAGE_COL_GAP * width:
            cols.append(x)
    if len(cols) < 2:
        return []
    cuts = [c - 2.0 for c in cols[1:]]
    edges = [-1e6] + cuts + [1e6]
    return list(zip(edges[:-1], edges[1:]))


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


def paragraphs(lines: list[dict], body_size: float, margin: float,
               italic_quotes: bool = True) -> list[dict]:
    """
    Lines gathered into paragraphs.

    A paragraph breaks when the italic flag flips (a quoted verse against the
    prose around it), when the size changes, on a wider-than-usual vertical gap,
    or -- for body text only -- when a line is indented past the margin, which is
    how a new paragraph announces itself. Quoted blocks deliberately do NOT break
    on indent: a verse's own lines are indented relative to its leading number,
    and splitting there would cut a couplet in half.

    `italic_quotes` is what the italic face MEANS in this book. In every book
    before In Search of the True Guru it set the quoted verse; there it sets the
    chapter titles and the Punjabi terms (Naam, sewa, Amrit), and a line whose
    ink is mostly one of those -- the short last line of a paragraph -- was cut
    off as a one-word "quote", 236 times. With it False the italic is still
    recorded on the paragraph and decides nothing.
    """
    lines = [ln for ln in lines if ln["text"]]
    if not lines:
        return []
    gaps = [lines[i - 1]["y"] - lines[i]["y"] for i in range(1, len(lines))]
    normal = statistics.median(gaps) if gaps else 0.0
    out: list[dict] = []
    cur = None
    for i, ln in enumerate(lines):
        gap = (lines[i - 1]["y"] - ln["y"]) if i else 0.0
        big = ln["size"] >= body_size * 1.15 and len(ln["text"]) < 80
        if italic_quotes:
            style = "quote" if ln["italic"] else ("heading" if big else "body")
        else:
            # An italic line that OPENS a paragraph -- first on the page, after
            # a wide gap, or indented -- and is shaped like a title is one:
            # "Charitable Acts", "The Holocaust". Read as body it ran into the
            # poem beneath it. An italic line inside a paragraph is a Punjabi
            # term's, and decides nothing.
            opens = cur is None or (normal and gap > normal * LEADING_BREAK) or ln["x0"] > margin + INDENT
            titled = (ln["italic"] and opens and len(ln["text"]) < 80
                      and not re.search(r"[.,;:]$", ln["text"]))
            style = "heading" if big or titled else "body"
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


HEAD_BAND = 2          # lines from the top or bottom of a page that may be a running head
HEAD_MAX = 60          # a running head is short
HEAD_SHARE = 0.30      # and stands on at least this share of the pages
# The printed number sits beside the head, and a book separates them with
# either a space ("NATURAL MEDITATION 65") or a slash ("SUNDRI/41"). Without the
# slash form every page's head is a different string, none recurs, and the whole
# running-head test never fires for that book.
HEAD_NUM = re.compile(r"^\s*[0-9ivxlc]{1,6}\s*/\s*|\s*/\s*[0-9ivxlc]{1,6}\s*$"
                      r"|^\s*[0-9ivxlc]{1,6}\s+|\s+[0-9ivxlc]{1,6}\s*$", re.I)
STAMP_MAX = 40         # a watermark is short
STAMP_SHARE = 0.60     # and is stamped on most pages, wherever it lands
HEAD_REPEATS = 3       # a numbered chapter head must still recur this many times


def running_heads(per_page: list[list[dict]]) -> tuple:
    """
    The lines that are furniture rather than text: `(texts, offset)`.

    A running head is recognised only at the very top or bottom of its page, and
    only when short — either condition alone would take real text with it. Two
    things then mark it, because books do it two ways:

      it recurs    `WWW.AKJ.ORG` stands on every page of the AKJ scans. The
                   printed number is stripped before counting, so `NATURAL
                   MEDITATION 65` and `... 67` count as one head.
      it is numbered   a head that names the current chapter changes too often
                   to recur — but it carries the printed page number, and that
                   number tracks the PDF's own index at a fixed offset through
                   the whole book. Finding the offset identifies the head on
                   every page, including the chapters that appear on two pages.

    Bau Ji's essays have neither: their header IS the `L127.1` marker, which is
    captured and dropped before this, so both come back empty for him.
    """
    seen: dict = {}
    offsets: dict = {}
    stamps: dict = {}
    pages = 0
    for n, lines in enumerate(per_page, start=1):
        if not lines:
            continue
        pages += 1
        # a watermark is not at an edge -- it is stamped wherever it lands, so
        # it is found by recurring on nearly every page instead
        for text in {ln["text"].strip() for ln in lines if len(ln["text"].strip()) <= STAMP_MAX}:
            if text:
                stamps[text] = stamps.get(text, 0) + 1
        edge = lines[:HEAD_BAND] + lines[-HEAD_BAND:]
        for text in {ln["text"].strip() for ln in edge if len(ln["text"].strip()) <= HEAD_MAX}:
            bare = HEAD_NUM.sub("", text).strip()
            if bare:
                seen[bare] = seen.get(bare, 0) + 1
            for m in re.finditer(r"[0-9]{1,4}", text):
                offsets[n - int(m.group())] = offsets.get(n - int(m.group()), 0) + 1
    floor = max(2, int(pages * HEAD_SHARE))
    texts = {t for t, n in seen.items() if n >= floor}
    best = max(offsets.items(), key=lambda kv: kv[1], default=(None, 0))
    mark = max(2, int(pages * STAMP_SHARE))
    # longest first, so a stamp containing another is removed whole
    marks = sorted((t for t, n in stamps.items() if n >= mark), key=len, reverse=True)
    # A chapter head recurs over its own chapter, not the book, so it never
    # reaches `floor`. Carrying the page number earns it a much lower bar --
    # but NOT no bar: "2. Where does it come from?" is a numbered list item at
    # the top of page 2, and dropping it would lose a real line of the essay.
    repeated = {t for t, n in seen.items() if n >= HEAD_REPEATS}
    return texts, (best[0] if best[1] >= floor else None), marks, repeated


def read_pdf(path: str, furniture: bool = False, italic_quotes: bool = True,
             page_columns_on: bool = False, overprint: int = 0,
             drop: list | None = None, drop_above: float | None = None,
             shadow: bool = False, drop_legacy: bool = False) -> dict:
    """
    One PDF as pages of paragraphs.

    `furniture` strips running heads, printed page numbers and watermarks. It is
    OFF by default and that is not a judgement about whether it helps: Bau Ji's
    corpus is already built and shipped from this reader, and these filters move
    its output (his essay titles double as running heads). A published corpus
    does not get to shift underneath a change made for a different book. Pass it
    for a new author -- the AKJ scans carry `www.AKJ.Org` on all 373 pages.

    @returns {"path", "body_size", "pages": [{"page", "marker", "spread", "paragraphs"}]}
    where a paragraph is {"text", "style", "italic", "x0", "size"}.

    `page_columns_on` decides columns page by page (page_columns) where the
    whole-document profile finds none, and `overprint` (the book's number of
    fake-bold copies) collapses them (collapse_overprint); both off by default for the same reason as
    `furniture`. `drop` is a book's own list of line patterns to remove --
    a running head or a page number the generic furniture filter does not
    recognise ("•  12  •"); a run whose whole text matches one is dropped, and `drop_above` drops every
    run printed higher on the page than that y (a chapter title as running head).
    """
    reader = PdfReader(path)
    per_page: list[dict] = []
    for n, page in enumerate(reader.pages, start=1):
        try:
            runs = page_runs(page)
            if overprint:
                collapse_overprint(runs, overprint)
            if shadow or overprint:
                drop_shadows(runs)
            if drop:
                runs = [r for r in runs if not any(re.fullmatch(d, r["text"].strip()) for d in drop)]
            if drop_above is not None:
                runs = [r for r in runs if r["y"] <= drop_above]
            if drop_legacy:
                runs = [r for r in runs if not legacy_font_run(r["text"])]
            per_page.append({"page": n, "runs": runs, "width": float(page.mediabox.width)})
        except Exception as exc:              # a damaged page must not lose the book
            per_page.append({"page": n, "runs": [], "error": str(exc)})
    bands = columns([p["runs"] for p in per_page])

    sizes: list[float] = []
    raw: list[dict] = []
    for p in per_page:
        runs = p["runs"]
        here = bands or (page_columns(runs, p.get("width", 600.0)) if page_columns_on else [])
        cols = ([group_lines([r for r in runs if lo <= r["x"] < hi]) for lo, hi in here]
                if here else [group_lines(runs)])
        every = [ln for col in cols for ln in col]
        raw.append({"page": p["page"], "cols": cols, "spread": is_spread(every),
                    "columns": len(here), **({"error": p["error"]} if p.get("error") else {})})
        sizes.extend(ln["size"] for ln in every if len(ln["text"]) > 20)
    body = statistics.median(sizes) if sizes else 10.0
    heads, head_offset, stamps, repeated = running_heads(
        [[ln for col in pg.get("cols", []) for ln in col] for pg in raw]) if furniture \
        else (set(), None, [], set())
    # A head is usually its own line and is dropped whole. Sometimes the scan
    # runs it into the first line of the body -- "40/SUNDRI wrought havoc among
    # the Sikhs" -- and then the line is too long to look like a head at all. So
    # a head is ALSO removed as a string, but only where it is glued to a page
    # number at the very start or end of the line, which is the one shape that
    # cannot be a sentence.
    alts = "|".join(sorted((re.escape(h) for h in heads if h), key=len, reverse=True))
    head_glued = re.compile(
        r"^\s*[0-9ivxlc]{1,6}\s*/\s*(?:%s)|(?:%s)\s*/\s*[0-9ivxlc]{1,6}\s*$"
        % (alts, alts), re.I) if alts else None

    pages = []
    for pg in raw:
        marker, out = None, []
        edge = {id(ln) for col in pg.get("cols", []) for ln in (col[:HEAD_BAND] + col[-HEAD_BAND:])}
        printed = str(pg["page"] - head_offset) if head_offset is not None else None
        for col in pg.get("cols", []):
            kept = []
            for ln in col:
                text = ln["text"].strip()
                # FIRST, before any furniture is discarded: the marker is an
                # essay's only in-document identifier, and it stands exactly
                # where a running head does. Dropping it as furniture would
                # lose it silently.
                found = MARKER.findall(text.replace(" ", ""))
                if found and len(text) < 32:
                    essay, page_no = found[-1]   # the English half's marker wins
                    marker = marker or f"L{int(essay)}.{int(page_no)}"
                    continue
                # A watermark is removed as a STRING, not as a line: these
                # scans stamp it wherever it lands, and it arrives glued onto
                # real text as often as alone ("Super-Natural Power Of
                # Amritwww.AKJ.Org").
                for s in stamps:
                    if s in text:
                        text = text.replace(s, " ")
                if head_glued is not None:
                    text = head_glued.sub(" ", text)
                ln["text"] = text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue
                if HEAD_NUM.sub("", text).strip() in heads:
                    continue                     # a running head or foot
                # the same line named by its printed page number instead, but
                # only where the rest of it is a head that recurs
                if (printed and id(ln) in edge and len(text) <= HEAD_MAX
                        and HEAD_NUM.sub("", text).strip() in repeated
                        and re.search(r"(?<![0-9])%s(?![0-9])" % re.escape(printed), text)):
                    continue
                if PAGE_NO.match(text):
                    continue
                if pg.get("spread") and ln["x0"] < SPREAD_X:
                    continue                     # stray ink from the scanned Punjabi half
                kept.append(ln)
            paras = paragraphs(kept, body, left_margin(kept), italic_quotes)
            if drop_legacy:
                # the legacy words a line of English carries (strip_legacy_words)
                paras = [q for q in ({**q, "text": strip_legacy_words(q["text"])} for q in paras) if q["text"]]
            out.extend(paras)
        pages.append({
            "page": pg["page"], "marker": marker, "spread": bool(pg.get("spread")),
            "columns": pg.get("columns", 0), "paragraphs": out,
            **({"error": pg["error"]} if pg.get("error") else {}),
        })
    return {"path": path, "body_size": body, "pages": pages}
