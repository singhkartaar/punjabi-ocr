"""
The grid reader: a notation's rows, columns and cells from the ink of the page.

A Bhatkhande grid is rows of swaras over rows of sung syllables, cut into
vibhags by bars, one avartan of the taal (or two) to a row. The page OCR
that named the shabad is no use inside the grid -- Tesseract glues bars to
letters and drops sparse rows -- so this reads the ink itself:

  1. bars: thick tall strokes inside the region are the vibhag boundaries,
     and are erased before anything else looks at the ink;
  2. row bands: the horizontal profile of what is left; a band holding two
     rows is cut at its thinnest line, a sliver (dots, an underline) joins
     the row it belongs to;
  3. rows come in pairs -- a swar row and its bol row stand close, pairs
     stand apart -- so the pairing gives every band its role, in the order
     the book prints (style.swar_row); a lone band of small marks is a
     marker row;
  4. the components of a row are clustered into cells by the gaps between
     them, and the cells of a row are read by Tesseract in one go, pasted
     into a strip with wide gaps so that each comes back as its own word;
  5. the bars and the taal's vibhag sizes give the grid its columns; a row
     that starts or ends inside a vibhag (the mukhda) is placed by the
     columns the full rows established; two letters in one cell are two
     notes in one matra; a column nothing fills is an unread beat; the bol
     cells go to the beat whose swara stands over them;
  6. a swar cell is its letter and its marks from the components around the
     letter: a dot above or below for the octave, a line beneath for komal,
     a stroke above for tivra. Glyph ambiguity is resolved by the row's
     role, never by the glyph.

Everything here is measured, not guessed: a cell nothing could read is
`notes: null` with what was seen in `raw`, never a silent Sa.
"""
from __future__ import annotations

import re

from lib.notation_text import clean_bol, is_marker, swara_token
from lib.notation_vocab import taal_from_markers, taal_info
from lib.ocr_grid import components, erase

# a component this small, this round, next to a letter, is a dot
DOT_AREA = 0.06          # of the letter's height squared
DOT_SIZE = 0.3           # neither side longer than this share of the letter's height
DOT_ASPECT = (0.45, 2.2)
# an underline: thin, and most of the letter's width
KOMAL_H = 0.22
KOMAL_W = 0.55
# a tivra stroke: thin, tall, above the letter
TIVRA_W = 0.35
TIVRA_H = 0.28
# a dash: a wide thin component on its own in the cell
DASH_ASPECT = 2.2
DASH_H = 0.45
BAR_MAX_W = 20
STRIP_PSM = 7
STRIP_GAP = 90           # px between cells in the strip Tesseract reads

_SWAR_LETTERS = set("ਸਰਗਮਪਧਨ")
_SWAR_SIGNS = set("ੇੀੁੂੰਂ਼ੑ")
_HALANT = "੍"
_BOL_SIGNS = set("ਾਿੈੋੌੱ")
_LETTER = re.compile("[ਅ-ਹਲ਼ਸ਼ਖ਼ਗ਼ਜ਼ੜਫ਼]")
_LABELS = ("ਸਰਗਮ", "ਸ਼ਬਦ", "ਸਬਦ", "ਮਾਤਰੇ", "ਮਾਤਰਾ", "ਤਾਲ", "ਚਿੰਨ੍ਹ", "ਚਿੰਨ", "ਸੁਰ", "ਬੋਲ")
_DIGITS = re.compile("^[0-9੦-੯]{1,2}$")


def _median(xs, default=0):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else default


# ---- cells ---------------------------------------------------------------------

def cells_from_components(comps: list[dict], gap: int) -> list[dict]:
    """
    Components of one band clustered into cells by the horizontal gap
    between them: [{"x0","y0","x1","y1","cx","comps":[...]}] left to right.
    A component that overlaps a cell's x-range (a dot above a letter) joins it.
    """
    out: list[dict] = []
    for c in sorted(comps, key=lambda c: c["x0"]):
        if out and c["x0"] <= out[-1]["x1"] + gap:
            cell = out[-1]
            cell["x1"] = max(cell["x1"], c["x1"])
            cell["y0"] = min(cell["y0"], c["y0"])
            cell["y1"] = max(cell["y1"], c["y1"])
            cell["comps"].append(c)
        else:
            out.append({"x0": c["x0"], "y0": c["y0"], "x1": c["x1"], "y1": c["y1"], "comps": [c]})
    for cell in out:
        cell["cx"] = (cell["x0"] + cell["x1"]) / 2.0
        cell["w"], cell["h"] = cell["x1"] - cell["x0"], cell["y1"] - cell["y0"]
    return out


def marks_from_components(cell: dict) -> dict:
    """
    The letter of a swar cell and the marks around it:
    {"main": comp|None, "octave": -2..2, "komal": bool, "tivra": bool, "dash": bool, "dots": n}.
    The main component is the largest; a small round one above or below it
    is an octave dot, a thin wide one under it the komal line, a thin tall
    one over it the tivra stroke. A cell that is one wide thin component is
    a dash (the held mark of the swar row).
    """
    comps = cell["comps"]
    if not comps:
        return {"main": None, "octave": 0, "komal": False, "tivra": False, "dash": False, "dots": 0}
    main = max(comps, key=lambda c: c["area"])
    mh = max(1, main["h"])
    mw = max(1, main["w"])
    if len(comps) == 1 and main["w"] >= DASH_ASPECT * main["h"] and main["h"] <= DASH_H * mh * 3:
        return {"main": main, "octave": 0, "komal": False, "tivra": False, "dash": True, "dots": 0}
    above = below = 0
    komal = tivra = False
    for c in comps:
        if c is main:
            continue
        aspect = c["w"] / max(1, c["h"])
        small = c["area"] <= DOT_AREA * mh * mh and max(c["w"], c["h"]) <= DOT_SIZE * mh
        roundish = DOT_ASPECT[0] <= aspect <= DOT_ASPECT[1]
        overlaps_x = c["x1"] > main["x0"] - 0.2 * mw and c["x0"] < main["x1"] + 0.2 * mw
        if small and roundish and overlaps_x:
            if c["cy"] < main["y0"] + 0.15 * mh:
                above += 1
            elif c["cy"] > main["y1"] - 0.15 * mh:
                below += 1
            continue
        if c["y0"] >= main["y1"] - 0.15 * mh and c["h"] <= KOMAL_H * mh and c["w"] >= KOMAL_W * mw:
            komal = True
            continue
        if c["y1"] <= main["y0"] + 0.25 * mh and c["w"] <= TIVRA_W * mh and c["h"] >= TIVRA_H * mh and aspect < 1.0:
            tivra = True
            continue
    octave = 0
    if above and not below:
        octave = min(2, above)
    elif below and not above:
        octave = -min(2, below)
    return {"main": main, "octave": octave, "komal": komal, "tivra": tivra, "dash": False, "dots": above + below}


# ---- beats -------------------------------------------------------------------

def fit_matras(counts: list[int], vibhag: list[int]) -> tuple[int, int]:
    """
    Which vibhag a row's first segment is, given how many cells each segment
    holds: (offset into `vibhag`, matra_from). A row that starts mid-cycle
    (a mukhda) has a short first segment; the offset that best matches the
    cell counts wins, ties to the start of the cycle.
    """
    if not vibhag:
        return 0, 1
    best, best_cost = 0, None
    for off in range(len(vibhag)):
        cost = 0
        for i, c in enumerate(counts):
            size = vibhag[(off + i) % len(vibhag)]
            cost += abs(c - size) if i else max(0, c - size)
        if best_cost is None or cost < best_cost:
            best, best_cost = off, cost
    matra_from = 1 + sum(vibhag[:best])
    first_size = vibhag[best]
    if counts and counts[0] < first_size:
        matra_from += first_size - counts[0]          # the first segment is partial: the row begins late
    return best, matra_from


def group_cells(cells: list[dict], n: int, pitch: float) -> list[list[dict]]:
    """
    `cells` of one segment as `n` beats: when there are more cells than
    beats the closest pairs merge (two notes in one matra); when fewer, the
    widest gaps get an empty beat (an unread cell), the rest at the end.
    """
    groups = [[c] for c in cells]
    while len(groups) > n > 0:
        gaps = [(groups[i + 1][0]["x0"] - groups[i][-1]["x1"], i) for i in range(len(groups) - 1)]
        gap, i = min(gaps)
        groups[i:i + 2] = [groups[i] + groups[i + 1]]
    missing = n - len(groups)
    if missing > 0 and groups:
        gaps = sorted(((groups[i + 1][0]["x0"] - groups[i][-1]["x1"], i + 1) for i in range(len(groups) - 1)), reverse=True)
        inserts = [at for gap, at in gaps[:missing] if gap > 0.6 * pitch]
        while len(inserts) < missing:
            inserts.append(len(groups))
        for at in sorted(inserts, reverse=True):
            groups.insert(at, [])
    return groups


def place_cells(cells: list[dict], columns: list[float]) -> list[list[dict]]:
    """
    `cells` of one segment placed on its `columns` (x centres, one a matra):
    each cell goes to the nearest column, so a row that begins or ends
    inside the vibhag keeps its matras. Returns one list per column.
    """
    out: list[list[dict]] = [[] for _ in columns]
    if not columns:
        return out
    for c in cells:
        k = min(range(len(columns)), key=lambda i: abs(columns[i] - c["cx"]))
        out[k].append(c)
    return out


def gap_boundaries(swar_rows: list[dict], pitch: float) -> list[int]:
    """
    Where a book prints no bars, the vibhags show as wider gaps between the
    cells: the midpoints of the gaps at least 1.7 times the usual pitch, as
    the fullest rows agree on them.
    """
    votes: list[int] = []
    rows = sorted(swar_rows, key=lambda r: -len(r["cells"]))[:4]
    for r in rows:
        cells = r["cells"]
        for k in range(len(cells) - 1):
            gap = cells[k + 1]["cx"] - cells[k]["cx"]
            if gap >= 1.7 * pitch:
                votes.append(int((cells[k]["x1"] + cells[k + 1]["x0"]) / 2))
    votes.sort()
    out: list[int] = []
    for x in votes:
        if out and x - out[-1] <= 0.8 * pitch:
            continue
        out.append(x)
    return out


def segments_from_bars(bar_xs: list[int], x0: int, x1: int, min_w: int) -> list[tuple[int, int]]:
    """(left, right) of each vibhag between the bars, edges included; slivers narrower than min_w are dropped."""
    edges = [x0] + [x for x in sorted(bar_xs) if x0 < x < x1] + [x1]
    out = []
    for a, b in zip(edges, edges[1:]):
        if b - a >= min_w:
            out.append((a, b))
    return out


def swar_letters(text: str, script: str = "gurmukhi") -> list[dict] | None:
    """
    The notes a swar cell's text spells: one letter, or two or three set
    side by side ("ਪਮ", "ਗਰ") for as many notes in the matra. None when any
    letter is not a swara. Vowel signs stay with their letter: the octave
    dots Tesseract reads as ੁ or ੰ become hints.
    """
    t = (text or "").strip()
    if not t:
        return None
    parts: list[str] = []
    for ch in t:
        if _LETTER.match(ch) or (ch.isalpha() and ch.isascii()):
            parts.append(ch)
        elif parts and (ch in _SWAR_SIGNS or ch in _BOL_SIGNS):
            parts[-1] += ch
        elif ch.isspace() or ch in "()|।[]!" or ch == _HALANT:
            continue
        else:
            return None
    if not parts or len(parts) > 3:
        return None
    notes = []
    for p in parts:
        got = swara_token(p, script)
        if not got or not got.get("swar"):
            return None
        notes.append(got)
    return notes


def pair_bands(bands: list[tuple[int, int]], first: str = "swar") -> list[str]:
    """
    The role of each band from how they stand: a swar row and its bol row
    are close, pairs are apart, so bands are paired by their gaps, the
    closest pairs first, and the first of a pair is `first`. A band left
    alone is a "lone" (a marker row, or a row whose partner is missing).
    """
    n = len(bands)
    if not n:
        return []
    heights = [b - a for a, b in bands]
    gaps = [bands[i + 1][0] - bands[i][1] for i in range(n - 1)]
    typical = _median(heights, 30)
    near = 0.6 * typical
    roles = ["lone"] * n
    other = "bol" if first == "swar" else "swar"
    for gap, i in sorted((g, i) for i, g in enumerate(gaps)):
        if gap > near:
            break
        if roles[i] == "lone" and roles[i + 1] == "lone":
            roles[i], roles[i + 1] = first, other
    return roles


def bar_components(ink, bbox: list[int], glyph_h: int) -> list[dict]:
    """
    The vibhag bars of a region: connected components of the ink that are
    tall, thin and on their own -- a letter's stem is part of a wider
    component, a bar is not. [{"x", "y0", "y1", "thick"}] sorted by x.
    """
    out = []
    for c in components(ink, bbox, min_area=20):
        if c["h"] >= 1.6 * glyph_h and c["w"] <= BAR_MAX_W and c["h"] >= 4 * c["w"]:
            out.append({"x": int(c["cx"]), "y0": c["y0"], "y1": c["y1"], "thick": c["w"]})
    out.sort(key=lambda r: r["x"])
    merged: list[dict] = []
    for r in out:
        if merged and abs(r["x"] - merged[-1]["x"]) <= 10:
            m = merged[-1]
            m["y0"], m["y1"], m["thick"] = min(m["y0"], r["y0"]), max(m["y1"], r["y1"]), max(m["thick"], r["thick"])
        else:
            merged.append(dict(r))
    return merged


# ---- reading one region ---------------------------------------------------------

def ocr_cells(page_img, cells: list[dict], engine, whitelist: str | None = None) -> list[tuple[str, float]]:
    """
    Every cell's text, from one Tesseract call: the cells are pasted into a
    strip with wide white gaps between them, so each comes back as its own
    word and the words map back by position. Cells nothing came back for
    are ("", 0.0).
    """
    import numpy as np
    if engine is None or page_img is None or not cells:
        return [("", 0.0) for _ in cells]
    pad = 6
    crops = []
    for c in cells:
        y0, y1 = max(0, c["y0"] - pad), min(page_img.shape[0], c["y1"] + pad)
        x0, x1 = max(0, c["x0"] - pad), min(page_img.shape[1], c["x1"] + pad)
        crops.append(page_img[y0:y1, x0:x1])
    h = max(cr.shape[0] for cr in crops) + 2 * pad
    w = sum(cr.shape[1] for cr in crops) + STRIP_GAP * (len(crops) + 1)
    strip = np.full((h, w), 255, dtype=page_img.dtype)
    spans = []
    x = STRIP_GAP
    for cr in crops:
        y = (h - cr.shape[0]) // 2
        strip[y:y + cr.shape[0], x:x + cr.shape[1]] = cr
        spans.append((x, x + cr.shape[1]))
        x += cr.shape[1] + STRIP_GAP
    got = engine.recognise_region(strip, [0, 0, w, h], "pa", psm=STRIP_PSM, whitelist=whitelist, pad=0)
    words = [wd for ln in got for wd in ln.get("words", [])] or [
        {"bbox": ln["bbox"], "text": ln["text"], "conf": ln.get("conf", 0.0)} for ln in got]
    out: list[tuple[str, float]] = [("", 0.0) for _ in cells]
    for wd in words:
        wx0, wx1 = wd["bbox"][0], wd["bbox"][2]
        wc = (wx0 + wx1) / 2.0
        k = min(range(len(spans)), key=lambda i: abs((spans[i][0] + spans[i][1]) / 2.0 - wc))
        # a word that straddles two cells belongs to the one holding most of it
        best, best_inter = k, -1
        for i, (sx0, sx1) in enumerate(spans):
            inter = min(sx1, wx1) - max(sx0, wx0)
            if inter > best_inter:
                best, best_inter = i, inter
        t, cf = out[best]
        out[best] = ((t + " " + wd["text"]).strip(), max(cf, float(wd.get("conf", 0.0))))
    return out


def _marker_glyph(cell: dict, ink) -> str | None:
    """
    × or 0 by shape, from where the ink lies inside the glyph's box: a
    cross is inked on its diagonals and at its centre and open at the
    middle of its sides; a ring is inked at the middle of every side and
    open at its centre. Anything else is left to the OCR.
    """
    if ink is None or not cell["comps"]:
        return None
    m = max(cell["comps"], key=lambda c: c["area"])
    w, h = m["w"], m["h"]
    if w < 6 or h < 6 or w > 1.7 * h or h > 1.7 * w:
        return None

    def inked(fx, fy):
        x, y = int(m["x0"] + fx * w), int(m["y0"] + fy * h)
        win = ink[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        return win.size > 0 and (win > 0).mean() >= 0.3

    on = lambda pts: all(inked(fx, fy) for fx, fy in pts)          # noqa: E731
    off = lambda pts: not any(inked(fx, fy) for fx, fy in pts)     # noqa: E731
    if on([(0.2, 0.2), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2), (0.5, 0.5)]) and off([(0.5, 0.15), (0.5, 0.85), (0.15, 0.5), (0.85, 0.5)]):
        return "×"
    if on([(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]) and off([(0.5, 0.5), (0.35, 0.35), (0.65, 0.65)]):
        return "0"
    return None


def letters_in_cell(cell: dict) -> int:
    """How many letters the cell's ink can hold: its letter-sized components, a wide one counting for the letters it joins."""
    comps = [c for c in cell["comps"] if c["h"] >= 0.45 * max(c["h"] for c in cell["comps"])]
    main = max(cell["comps"], key=lambda c: c["area"])
    n = 0
    for c in comps:
        n += max(1, int(round(c["w"] / max(1.0, 0.95 * main["h"]))))
    return max(1, n)


def _pick_swar_text(cell: dict, first: tuple[str, float], second: tuple[str, float], script: str) -> tuple[str, float]:
    """Of two readings of a swar cell, the one that spells swaras fitting the cell; the first when neither or both do."""
    want = letters_in_cell(cell)
    def score(t):
        reads = swar_letters(t.strip().strip("|।!"), script) if t else None
        if reads is None:
            return 0
        return 2 if len(reads) == want else 1
    a, b = score(first[0]), score(second[0])
    if b > a:
        return second
    return first


def _read_swar_cell(cell: dict, text: str, conf: float, script: str) -> tuple[dict | None, str, float]:
    """(beat body, raw text, confidence) for one swar cell from its text and its components."""
    marks = marks_from_components(cell)
    raw = text
    if marks["dash"]:
        return {"ext": True}, raw or "-", 1.0
    t = (text or "").strip().strip("|।!")
    if t and set(t) <= set("-—–―_="):
        return {"ext": True}, raw, conf
    if t in ("×", "x", "X", "*", "∗"):
        return {"rest": True}, raw, conf
    reads = swar_letters(t, script) if t else None
    if reads is None and t:
        # a one-letter cell whose text has a letter too many: the OCR read a mark as a letter
        main = marks["main"]
        if main is not None and main["w"] <= 1.4 * max(1, main["h"]):
            kept = "".join(ch for ch in t if ch in _SWAR_LETTERS or ch in _SWAR_SIGNS or ch in _BOL_SIGNS)
            letters = [ch for ch in kept if ch in _SWAR_LETTERS]
            if len(letters) == 1:
                reads = swar_letters(kept, script)
    if reads is not None and len(reads) > 1:
        reads = reads[: letters_in_cell(cell)]        # the ink holds fewer letters than the OCR spelt: a mark was read as one
    if reads is None:
        return None, raw, conf
    plain = len(cell["comps"]) == 1          # nothing but the letter: the OCR's vowel-sign hint is all there is
    notes = []
    for read in reads:
        note = {"s": read["swar"]}
        octave = (marks["octave"] if len(reads) == 1 else 0) or (read.get("octave_hint") if plain or len(reads) > 1 else 0) or 0
        if octave:
            note["o"] = int(octave)
        if marks["komal"] and len(reads) == 1 and read["swar"] in ("R", "G", "D", "N"):
            note["k"] = True
        if marks["tivra"] and len(reads) == 1 and read["swar"] == "M":
            note["t"] = True
        if read.get("khatka"):
            note["kh"] = True
        notes.append(note)
    return {"notes": notes}, raw, conf


def _token_look(text: str) -> str:
    """swar | bol | marker | dash | other, by the shape of one token."""
    t = (text or "").strip().strip("|।!:.,")
    if not t:
        return "other"
    if set(t) <= set("-—–―_="):
        return "dash"
    if is_marker(t):
        return "marker"
    if set(t) <= set("ऽ਽S$5s;"):
        return "bol"
    letters = [ch for ch in t if _LETTER.match(ch)]
    signs = [ch for ch in t if not _LETTER.match(ch)]
    if letters and all(ch in _SWAR_LETTERS for ch in letters) and not any(ch in _BOL_SIGNS for ch in signs) and len(letters) <= 3:
        return "swar"
    if letters:
        return "bol"
    return "other"


def _bol_of(texts: list[str]) -> dict | None:
    g_parts, held = [], 0
    for t in texts:
        g, h = clean_bol(t)
        if g:
            g_parts.append(g)
        held += h
    if not g_parts and not held:
        return None
    return {"g": " ".join(g_parts), "h": held}


def _segment_columns(rows: list[dict], segs: list[tuple[int, int]], sizes: list[int]) -> list[list[float]]:
    """
    The x centre of every matra column of every segment, from the swar rows
    that fill a segment completely; a segment no row fills is spaced evenly.
    """
    out: list[list[float]] = []
    for (a, b), n in zip(segs, sizes):
        samples: list[list[float]] = []
        for r in rows:
            if r["role"] != "swar":
                continue
            cells = [c for c in r["cells"] if a <= c["cx"] < b]
            if len(cells) == n:
                samples.append([c["cx"] for c in cells])
        if samples:
            out.append([_median([s[k] for s in samples]) for k in range(n)])
        else:
            step = (b - a) / max(1, n)
            out.append([a + step * (k + 0.5) for k in range(n)])
    return out


def rows_from_components(comps: list[dict], glyph_h: int) -> list[dict]:
    """
    The rows of a grid from its components: the letter-sized ones (and the
    dashes, which are wide) cluster by their vertical centre; the small
    ones -- dots, underlines, ticks -- join the row nearest to them, and
    those far from every row (the marks of a marker row) make rows of their
    own. Each row: {"y0","y1" (all its ink), "ly0","ly1" (its letters),
    "comps"}, top to bottom.
    """
    letters = [c for c in comps if c["h"] >= 0.45 * glyph_h or (c["w"] >= 1.2 * glyph_h and c["h"] >= 0.08 * glyph_h)]
    small = [c for c in comps if c not in letters]
    rows: list[dict] = []
    for c in sorted(letters, key=lambda c: c["cy"]):
        if rows and c["cy"] - rows[-1]["_cys"][-1] <= 0.55 * glyph_h:
            rows[-1]["comps"].append(c)
            rows[-1]["_cys"].append(c["cy"])
        else:
            rows.append({"comps": [c], "_cys": [c["cy"]]})
    for r in rows:
        r["ly0"] = min(c["y0"] for c in r["comps"])
        r["ly1"] = max(c["y1"] for c in r["comps"])
    stray: list[dict] = []
    for c in small:
        best, dist = None, None
        for r in rows:
            d = 0 if r["ly0"] <= c["cy"] <= r["ly1"] else min(abs(c["cy"] - r["ly0"]), abs(c["cy"] - r["ly1"]))
            if dist is None or d < dist:
                best, dist = r, d
        if best is not None and dist <= 0.5 * glyph_h:
            best["comps"].append(c)
        else:
            stray.append(c)
    # marks with no row of their own nearby: a marker row (× 0 2 3), clustered the same way
    for c in sorted(stray, key=lambda c: c["cy"]):
        near = [r for r in rows if r.get("_marks") and abs(c["cy"] - r["_cys"][-1]) <= 0.55 * glyph_h]
        if near:
            near[-1]["comps"].append(c)
            near[-1]["_cys"].append(c["cy"])
        else:
            rows.append({"comps": [c], "_cys": [c["cy"]], "_marks": True, "ly0": c["y0"], "ly1": c["y1"]})
    out = []
    for r in rows:
        y0 = min(c["y0"] for c in r["comps"])
        y1 = max(c["y1"] for c in r["comps"])
        ly0 = min(c["y0"] for c in r["comps"]) if r.get("_marks") else r["ly0"]
        ly1 = max(c["y1"] for c in r["comps"]) if r.get("_marks") else r["ly1"]
        # a row whose glyphs are all short is a marker row (× 0 2 3), not a row of letters
        big = [c for c in r["comps"] if c in letters]
        short = big and len(big) <= 8 and all(c["h"] <= 0.62 * glyph_h and c["w"] <= 0.9 * glyph_h for c in big)
        out.append({"y0": y0, "y1": y1, "ly0": ly0, "ly1": ly1, "comps": r["comps"], "marks": bool(r.get("_marks")) or bool(short)})
    out.sort(key=lambda r: r["ly0"])
    return out


def read_region(ink, page_img, bbox: list[int], style: dict, taal_key: str | None, engine, page: int,
                body_h: int | None = None, script: str = "gurmukhi") -> dict:
    """
    One grid region of a page -> {"lines": [Line], "bbox": the ink's extent,
    "marker_rows": [{matra: glyph}], "quality": {...}, "flags": [...],
    "taal_inferred": key|None, "rows": [...]}.
    `ink` is the binarised page (255 on 0), `page_img` the grayscale page for
    the OCR, `bbox` the region in page coordinates.
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    flags: list[str] = []
    taal = taal_info(taal_key)
    vibhag = list(taal["vibhag"]) if taal and taal.get("vibhag") else []
    matras = int(taal["matras"]) if taal and taal.get("matras") else None
    glyph_h = int(0.7 * (body_h or 50))
    probe = [c["h"] for c in components(ink, [x0, y0, x1, y1], min_area=12) if 10 <= c["h"] <= 2.5 * glyph_h and c["w"] <= 3 * glyph_h]
    if len(probe) >= 8:
        glyph_h = int(_median(probe, glyph_h))          # the letters of this grid, not the page's prose

    # 1. the bars, erased before the rest looks
    rules = bar_components(ink, [x0, y0, x1, y1], glyph_h)
    long_bars = [v["x"] for v in rules]
    clean = erase(ink, v_rules=rules, pad=4)
    comps_all = [c for c in components(clean, [x0, y0, x1, y1], min_area=3) if c["h"] < 3 * glyph_h and c["w"] < 12 * glyph_h]

    # 2. rows from the components; each row its cells
    rows = []
    for band in rows_from_components(comps_all, glyph_h):
        a, b = band["ly0"], band["ly1"]
        comps = band["comps"]
        gap = max(4, int(0.4 * _median([c["h"] for c in comps if c["h"] >= 0.3 * glyph_h], glyph_h)))
        cells = [c for c in cells_from_components(comps, gap)
                 if max(k["area"] for k in c["comps"]) >= 0.04 * glyph_h * glyph_h          # a speck
                 and not (c["w"] <= 4 and c["h"] >= 0.5 * glyph_h)]                          # a bar's remnant
        rows.append({"y0": band["y0"], "y1": band["y1"], "ly0": a, "ly1": b, "marks": band["marks"],
                     "bars": list(long_bars), "cells": cells, "words": [], "role": None, "texts": []})

    # 3. a marker row: at most a mark a vibhag, and the marks are crosses, rings or digits;
    #    a row of nothing but specks is noise; the rest alternate as the book prints them
    n_segs = len(long_bars) + 1
    first = "swar" if style.get("swar_row", "above") == "above" else "bol"
    other = "bol" if first == "swar" else "swar"
    # marker rows first: their shapes are the surest thing on the page. A letter
    # standing on the marker row's line (the first swara of a mukhda) is not a
    # marker, and gets a row of its own right after it.
    split: list[dict] = []
    for r in rows:
        r["role"] = None
        if not r["cells"]:
            r["role"] = "empty"
            split.append(r)
            continue
        shaped = [bool(_marker_glyph(c, ink)) for c in r["cells"]]
        short = all(c["h"] <= 0.8 * glyph_h for c in r["cells"])
        if len(r["cells"]) > n_segs + 2 and (r["ly1"] - r["ly0"]) < 0.55 * glyph_h and not any(shaped):
            r["role"] = "text"                      # a note printed small under the grid
            split.append(r)
            continue
        if len(r["cells"]) <= n_segs + 2 and (r["marks"] or sum(shaped) >= 0.5 * len(r["cells"]) or (short and len(r["cells"]) <= 4)):
            letters_ = [c for c, sh in zip(r["cells"], shaped) if not sh and c["h"] >= 0.8 * glyph_h]
            marks_ = [c for c in r["cells"] if c not in letters_]
            r["role"] = "marker"
            if letters_ and marks_:
                r["cells"] = marks_
                split.append(r)
                split.append({**r, "cells": letters_, "role": None, "marks": False, "texts": []})
                continue
        split.append(r)
    rows = split
    alt = getattr(engine, "alt", None)
    for r in rows:
        if not r["cells"] or r["role"] in ("empty", "text"):
            r["texts"] = [("", 0.0) for _ in r["cells"]]
            continue
        if r["role"] == "marker" and not r.get("_ocr"):
            pass
        r["texts"] = ocr_cells(page_img, r["cells"], engine)
        got = sum(1 for t, _ in r["texts"] if t)
        if got < 0.5 * len(r["cells"]) and alt is not None:
            second = ocr_cells(page_img, r["cells"], alt)
            if sum(1 for t, _ in second if t) > got:
                r["texts"] = second
                got = sum(1 for t, _ in second if t)
        if got < 0.5 * len(r["cells"]) and engine is not None and page_img is not None:
            # one cell at a time, as a single character
            texts = list(r["texts"])
            for k, c in enumerate(r["cells"]):
                if texts[k][0]:
                    continue
                one = engine.recognise_region(page_img, [c["x0"], c["y0"], c["x1"], c["y1"]], "pa", psm=10, pad=6)
                t = " ".join(ln["text"] for ln in one).strip()
                if t:
                    texts[k] = (t, float(_median([w.get("conf", 0.0) for ln in one for w in ln.get("words", [])], 0.0)))
            r["texts"] = texts
    # a label column (ਸਰਗਮ / ਸ਼ਬਦ / ਤਾਲ ਚਿੰਨ੍ਹ down the left) is the book's, not the music's:
    # the leading segments whose cells are mostly those words go, bars and all
    edges = [x0] + sorted(long_bars)
    while len(edges) >= 2:
        a_, b_ = edges[0], edges[1]
        cells_in = [(r, [t for c, (t, _) in zip(r["cells"], r["texts"]) if a_ <= c["cx"] < b_]) for r in rows]
        texts = [t for _, ts in cells_in for t in ts]
        labelled = sum(1 for t in texts if any(lbl in t for lbl in _LABELS))
        if not texts or labelled < max(2, 0.4 * len(texts)):
            break
        for r in rows:
            keep = [k for k, c in enumerate(r["cells"]) if c["cx"] >= b_]
            r["cells"] = [r["cells"][k] for k in keep]
            r["texts"] = [r["texts"][k] for k in keep]
        long_bars = [x for x in long_bars if x > b_]
        for r in rows:
            r["bars"] = list(long_bars)
        edges = [b_] + sorted(long_bars)
    # a row of matra numbers (1 2 3 4 …) says where the columns are and is not read as music
    for r in rows:
        if r["role"] or not r["cells"]:
            continue
        toks = [t.strip() for t, _ in r["texts"] if t.strip()]
        if len(toks) >= 3 and sum(1 for t in toks if _DIGITS.match(t)) >= 0.7 * len(toks):
            r["role"] = "matra"
    most = max((len(r["cells"]) for r in rows if r["role"] not in ("marker", "matra")), default=0)
    for i, r in enumerate(rows):
        if r["role"]:
            continue
        # a sparse row of letters standing just over a much fuller row: the kan (grace) notes of that row
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if (nxt and nxt["role"] is None and r["cells"] and len(r["cells"]) <= 0.5 * most and len(nxt["cells"]) >= 2 * len(r["cells"])
                and nxt["ly0"] - r["ly1"] <= 0.8 * glyph_h):
            r["role"] = "kan"
    # the rest alternate as the book prints them; a row whose tokens say otherwise
    # loudly (four or more, nearly all of one kind) resets the alternation from
    # itself, because one stray row must not turn every row after it around
    expect = first
    for r in rows:
        if r["role"]:
            continue
        looks = [_token_look(t) for t, _ in r["texts"] if t]
        swarish = sum(1 for look in looks if look in ("swar", "dash"))
        bolish = sum(1 for look in looks if look == "bol")
        loud = None
        if len(looks) >= 4 and swarish >= 0.8 * len(looks) and any(look == "dash" for look in looks):
            loud = "swar"
        elif len(looks) >= 4 and bolish >= 0.8 * len(looks):
            loud = "bol"
        if loud and loud != expect:
            if "style-contradiction" not in flags:
                flags.append("style-contradiction")
            expect = loud
        r["role"] = expect
        expect = other if expect == first else first

    # 4. a second reading of the swar rows: whichever model spells swaras that fit the cell
    if alt is not None:
        for r in rows:
            if r["role"] in ("swar", "kan") and r["cells"]:
                second = ocr_cells(page_img, r["cells"], alt)
                r["texts"] = [_pick_swar_text(c, a_, b_, script) for c, a_, b_ in zip(r["cells"], r["texts"], second)]

    # 5. the columns of the grid: the bars, the taal's sizes, the full rows' centres
    swar_rows = [r for r in rows if r["role"] == "swar" and r["cells"]]
    kan_of: dict[int, list[tuple[float, str]]] = {}          # index of the swar row -> [(x centre, letter)]
    for i, r in enumerate(rows):
        if r["role"] == "kan" and i + 1 < len(rows) and rows[i + 1]["role"] == "swar":
            kan_of[i + 1] = [(c["cx"], t) for c, (t, _) in zip(r["cells"], r["texts"]) if t]
    lines: list[dict] = []
    quality = {"cells": 0, "unknown": 0, "confs": [], "marks": {"dot_above": 0, "dot_below": 0, "underline": 0, "tick": 0}}
    marker_rows = [r for r in rows if r["role"] == "marker"]
    columns: list[list[float]] = []
    if swar_rows:
        all_bars = sorted(set(x for r in swar_rows for x in r["bars"]))
        pitch = _median([r["cells"][k + 1]["cx"] - r["cells"][k]["cx"] for r in swar_rows for k in range(len(r["cells"]) - 1)], glyph_h * 1.5)
        if not all_bars:
            # a bar-less grid: the vibhags are the wide gaps between the cells of the fullest rows
            all_bars = gap_boundaries(swar_rows, pitch)
        segs = segments_from_bars(all_bars, x0, x1, min_w=int(0.6 * pitch))

        def filled(seg):
            return any(any(seg[0] <= c["cx"] < seg[1] for c in r["cells"]) for r in swar_rows)

        while segs and not filled(segs[0]):
            segs.pop(0)
        while segs and not filled(segs[-1]):
            segs.pop()
        counts_full = [max((sum(1 for c in r["cells"] if a <= c["cx"] < b) for r in swar_rows), default=0) for a, b in segs]
        if vibhag:
            off, _ = fit_matras(counts_full, vibhag)
            sizes = [vibhag[(off + i) % len(vibhag)] for i in range(len(segs))]
            start_matra = [1 + sum(vibhag[:off])]
        else:
            sizes = [max(1, c) for c in counts_full]
            start_matra = [1]
        for k in range(1, len(segs)):
            start_matra.append(start_matra[-1] + sizes[k - 1])
        columns = _segment_columns(rows, segs, sizes)

        used = set()
        for i, r in enumerate(rows):
            if r["role"] != "swar" or not r["cells"] or i in used:
                continue
            partner = None
            for j in ((i + 1, i - 1) if first == "swar" else (i - 1, i + 1)):
                if 0 <= j < len(rows) and rows[j]["role"] == "bol" and j not in used:
                    partner = j
                    break
            used.add(i)
            if partner is not None:
                used.add(partner)
            bol_cells = rows[partner]["cells"] if partner is not None else []
            bol_texts = rows[partner]["texts"] if partner is not None else []
            cell_text = {id(c): t for c, t in zip(r["cells"], r["texts"])}
            placed: list[tuple[int, int, list[dict]]] = []
            for si, (a, b) in enumerate(segs):
                cells = [c for c in r["cells"] if a <= c["cx"] < b]
                for k, group in enumerate(place_cells(cells, columns[si])):
                    placed.append((si, k, group))
            first_k = next((n for n, (_, _, g) in enumerate(placed) if g), None)
            last_k = next((n for n in range(len(placed) - 1, -1, -1) if placed[n][2]), None)
            if first_k is None:
                continue
            placed = placed[first_k:last_k + 1]
            si0, k0, _ = placed[0]
            line_bbox = [x0, r["y0"], x1, rows[partner]["y1"] if partner is not None else r["y1"]]
            state = {"beats": [], "raw_swar": [], "raw_bol": [], "avartan": 1}
            m = (start_matra[si0] + k0) if vibhag else 1
            if matras:
                m = (m - 1) % matras + 1

            def flush():
                beats = state["beats"]
                if not beats:
                    return
                line = {"kind": "avartan" if matras else "free", "page": page, "bbox": list(line_bbox), "beats": beats,
                        "raw": {"swar": " ".join(state["raw_swar"]), "bol": " ".join(state["raw_bol"]),
                                "conf": round(_median([b["c"] for b in beats if isinstance(b.get("c"), float)], 1.0), 2)}}
                if matras:
                    line["avartan"] = state["avartan"]
                    if beats[0].get("m", 1) != 1:
                        line["matra_from"] = beats[0]["m"]
                lines.append(line)
                state["beats"], state["raw_swar"], state["raw_bol"] = [], [], []

            for si, k, group in placed:
                if matras and m > matras:
                    flush()
                    m = 1
                    state["avartan"] += 1
                beat: dict = {}
                if matras:
                    beat["m"] = m
                if not group:
                    beat.update({"notes": None, "raw": "?", "c": 0.0})
                    quality["unknown"] += 1
                    if "empty-beat" not in flags:
                        flags.append("empty-beat")
                    state["raw_swar"].append("?")
                else:
                    notes: list[dict] = []
                    raws, confs = [], []
                    kinds = set()
                    for cell in group:
                        text, conf = cell_text.get(id(cell), ("", 0.0))
                        body, raw, conf = _read_swar_cell(cell, text, conf, script)
                        raws.append(raw)
                        confs.append(conf)
                        mk = marks_from_components(cell)
                        quality["marks"]["dot_above"] += 1 if mk["octave"] > 0 else 0
                        quality["marks"]["dot_below"] += 1 if mk["octave"] < 0 else 0
                        quality["marks"]["underline"] += 1 if mk["komal"] else 0
                        quality["marks"]["tick"] += 1 if mk["tivra"] else 0
                        if body is None:
                            kinds.add("unknown")
                        elif "notes" in body:
                            notes.extend(body["notes"])
                        elif body.get("ext"):
                            if notes:
                                notes[-1]["len"] = notes[-1].get("len", 1) + 1
                            else:
                                kinds.add("ext")
                        else:
                            kinds.add("rest")
                    conf = _median([c for c in confs if c], 0.0)
                    state["raw_swar"].append("".join(raws) if len(group) > 1 else (raws[0] or "·"))
                    if notes:
                        for kx, kt in kan_of.get(i, []):
                            if abs(kx - columns[si][k]) <= 0.5 * pitch:
                                kread = swar_letters(kt.strip(), script)
                                if kread:
                                    kan = {"s": kread[0]["swar"]}
                                    if kread[0].get("octave_hint"):
                                        kan["o"] = int(kread[0]["octave_hint"])
                                    notes[0]["kan"] = kan
                                break
                        beat["notes"] = notes
                        total = sum(n_.get("len", 1) for n_ in notes)
                        if total > 1:
                            beat["div"] = total
                        if "unknown" in kinds:
                            beat["raw"] = " ".join(raws)
                            if "unread-cell" not in flags:
                                flags.append("unread-cell")
                        if conf < 0.6:
                            beat["c"] = round(conf, 2)
                    elif kinds == {"ext"}:
                        beat["ext"] = True
                    elif kinds == {"rest"}:
                        beat["rest"] = True
                    else:
                        beat.update({"notes": None, "raw": " ".join(raws) or "?", "c": round(conf, 2)})
                        quality["unknown"] += 1
                        if "unread-cell" not in flags:
                            flags.append("unread-cell")
                    quality["confs"].append(conf)
                if bol_cells:
                    cx = columns[si][k]
                    half = pitch * 0.5
                    under = [n_ for n_, c in enumerate(bol_cells) if abs(c["cx"] - cx) <= half or (c["x0"] < cx < c["x1"])]
                    bol = _bol_of([bol_texts[n_][0] for n_ in under if bol_texts[n_][0]])
                    if bol is not None:
                        beat["bol"] = bol
                        state["raw_bol"].append((bol["g"] or "") + "ऽ" * bol["h"])
                    else:
                        state["raw_bol"].append("·")
                quality["cells"] += 1
                state["beats"].append(beat)
                m += 1
            flush()

    # 6. the marker rows against the taal
    marker_checks: list[dict] = []
    centres = [cx for seg in columns for cx in seg]
    for mr in marker_rows:
        if not centres:
            break
        marks: dict[int, str] = {}
        for c, (text, _) in zip(mr["cells"], mr["texts"]):
            if max(k["area"] for k in c["comps"]) < 0.04 * glyph_h * glyph_h:
                continue
            glyph = _marker_glyph(c, ink) or (text.strip() if is_marker(text.strip()) else None)
            if not glyph or not centres:
                continue
            k = min(range(len(centres)), key=lambda i: abs(centres[i] - c["cx"]))
            marks[k + 1] = glyph
        if marks:
            marker_checks.append(marks)
    if marker_checks and matras and taal:
        for marks in marker_checks:
            sams = sorted(mtr for mtr, g in marks.items() if is_marker(g) == "sam")
            if any((s - 1) % matras + 1 != taal.get("sam", 1) for s in sams):
                if "taal-mismatch" not in flags:
                    flags.append("taal-mismatch")
    inferred = None
    if not matras and marker_checks and lines:
        inferred = taal_from_markers(len(lines[0]["beats"]), marker_checks[0])

    allcells = [c for r in rows for c in r["cells"]]
    extent = [x0, y0, x1, y1]
    if allcells:
        extent = [min(x0, min(c["x0"] for c in allcells)), min(y0, min(c["y0"] for c in allcells)),
                  max(x1, max(c["x1"] for c in allcells)), max(y1, max(c["y1"] for c in allcells))]
    confs = quality.pop("confs")
    quality["conf_mean"] = round(sum(confs) / len(confs), 3) if confs else 0.0
    return {"lines": lines, "bbox": extent, "marker_rows": marker_checks, "quality": quality, "flags": flags,
            "taal_inferred": inferred, "columns_dbg": columns, "bars_dbg": long_bars,
            "rows": [{"y0": r["y0"], "y1": r["y1"], "role": r["role"], "cells": len(r["cells"]),
                      "text": " ".join(t for t, _ in r["texts"])[:140]} for r in rows]}
