"""
The ink on a page that no recognised line covers.

Tesseract's page layout is right where it finds a line and wrong by omission:
on the Santhya's two-column pages it drops whole lines of the narrow verse
column, and nothing downstream can know, because a line that was never
segmented never appears anywhere. The parallel attempt to fix that replaced
the layout with a projection-profile line finder over every page and read
each box on its own; no line was missed and word accuracy fell from 95.8% to
79.7%. So the layout stays, and this module finds only what it missed: paint
every recognised box onto the page's ink, look at what remains, and hand the
merge the regions shaped like a line of text, for it to crop and read.

Everything here is a share of the page or a multiple of the typical line
height, never a pixel count, so a 200 dpi scan and a 400 dpi one are judged
alike. A region is a candidate when it is about a line tall, wide enough to be
a word or two, inked like text (not a rule, not a halftone, not dust), and
lies inside no recognised box; the merge then reads it and rejects what comes
back empty or as noise. Every region, taken or refused, is written to the
page's `_meta.coverage` with the reason, so a book's misses can be counted
and looked at rather than believed.
"""
from __future__ import annotations

from lib.ocr_zones import FOOTER_BAND, HEADER_BAND

PAD = 0.25             # of typical_h painted around every recognised box: Tesseract's boxes are tight,
                       # and a matra or a descender sticks out of them
MIN_ROW_INK_H = 0.75   # a row of a band is inked when its ink is at least this x typical_h px wide ...
MIN_ROW_INK_W = 0.02   # ... and at least this share of the band's width (a speck in a narrow column is no line)
GAP_ROWS = 0.15        # of typical_h: white rows bridged inside one run (the white between a headline and its matras)
MIN_H, MAX_H = 0.35, 2.5  # a run's height in typical_h: below is a matra sliver or a rule, above a block or a picture.
                          # Measured on the Santhya: a line cut at the valley under its headline is its x-height,
                          # 0.41-0.48 of the line (968 real lines were refused as "short" at 0.5); a sliver is
                          # under 0.2 and a rule under 0.1
VALLEY = 0.2           # a tall run is cut where its row ink falls under this share of its peak
MIN_W = 1.5            # a region's width in typical_h: "॥੧॥" passes, a stray dot does not
DENSITY = (0.06, 0.55) # ink / box area: text sits at 0.12-0.35; dust below, a bar or a halftone above
SOLID_ROW = 0.8        # a row inked over this share of the box's width is a rule, not glyphs ...
SOLID_SHARE = 0.5      # ... and a box with this share of such rows is rejected as one
LINE_OVERLAP_MAX = 0.2 # share of a region's area that may lie inside a recognised box
MIN_RESIDUAL = 0.003   # share of the page's ink left uncovered below which the page counts as covered
EDGE = 0.03            # share of page width ignored at either edge (binding shadow, scanner edge)
RULE_HALO = 0.012      # share of page width blanked around a vertical rule
RULE_Y_HALO = 0.15     # of typical_h blanked around the footnote rule's row
MAX_REGIONS = 40       # per page. The shape tests run before the cap, so a picture page is not the cost; a page
                       # whose layout dropped half its lines (an index page, a grey-banded scan) is, and it
                       # needs them all: 12 left 22 real lines unread on three pages of one book
MIN_LINES_FOR_H = 5    # fewer body lines than this and typical_h is taken from the page height instead
DEFAULT_H = 0.018      # share of page height: a body line on a 300 dpi octavo, when the page cannot say

REASONS = ("short", "tall", "narrow", "sparse", "dense", "rule", "overlaps-line",
           "empty", "no-letters", "junk", "capped", "low-agreement")


def ink_mask(img):
    """Dark pixels, the same cut lib/ocr_zones uses."""
    return img < 128


def typical_height(body_lines: list[dict], page_h: int) -> float:
    """The median body line height, or a share of the page when there are too few lines to say."""
    heights = sorted(ln["bbox"][3] - ln["bbox"][1] for ln in body_lines if ln.get("text", "").strip())
    if len(heights) < MIN_LINES_FOR_H:
        return page_h * DEFAULT_H
    return float(heights[len(heights) // 2])


def covered_mask(shape: tuple, lines: list[dict], pad_px: int):
    """True wherever a recognised box (any zone), padded, lies."""
    import numpy as np
    h, w = shape[:2]
    covered = np.zeros((h, w), dtype=bool)
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        covered[max(0, int(y0) - pad_px):min(h, int(y1) + pad_px), max(0, int(x0) - pad_px):min(w, int(x1) + pad_px)] = True
    return covered


def band_regions(residual, x_lo: int, x_hi: int, typical_h: float, col: int) -> list[dict]:
    """
    Regions of uncovered ink in one column band, each with its shape measured
    and a first verdict: "candidate", or the reason it is not a line of text.

    residual is the page's uncovered ink (bool, page rows x page columns);
    the band is the columns [x_lo, x_hi).
    """
    import numpy as np
    band = residual[:, x_lo:x_hi]
    if band.size == 0:
        return []
    row_ink = band.sum(axis=1)
    band_w = x_hi - x_lo
    inked = row_ink >= max(MIN_ROW_INK_H * typical_h, MIN_ROW_INK_W * band_w)
    bridge = max(1, int(round(GAP_ROWS * typical_h)))
    runs: list[tuple[int, int]] = []
    start = None
    gap = 0
    for y, on in enumerate(inked):
        if on:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > bridge:
                runs.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(inked) - gap))

    # a run taller than a line is cut where its ink thins: two lines whose
    # matras touch, or a line under a heading
    def cut(y0: int, y1: int) -> list[tuple[int, int]]:
        if y1 - y0 <= MAX_H * typical_h:
            return [(y0, y1)]
        prof = row_ink[y0:y1]
        peak = float(prof.max()) or 1.0
        pieces, s = [], y0
        for y in range(y0 + 1, y1 - 1):
            if row_ink[y] < VALLEY * peak and row_ink[y - 1] >= VALLEY * peak:
                pieces.append((s, y))
                s = y
        pieces.append((s, y1))
        return [(a, b) for a, b in pieces if b - a >= max(1, int(MIN_H * typical_h))] or [(y0, y1)]

    out: list[dict] = []
    for y0, y1 in runs:
        for a, b in cut(y0, y1):
            rows = band[a:b]
            cols = rows.any(axis=0)
            xs = np.flatnonzero(cols)
            if xs.size == 0:
                continue
            bx0, bx1 = x_lo + int(xs[0]), x_lo + int(xs[-1]) + 1
            box = [bx0, int(a), bx1, int(b)]
            h_rel = (b - a) / typical_h
            w_rel = (bx1 - bx0) / typical_h
            area = max(1, (b - a) * (bx1 - bx0))
            density = float(rows[:, xs[0]:xs[-1] + 1].sum()) / area
            solid = float((rows[:, xs[0]:xs[-1] + 1].sum(axis=1) >= SOLID_ROW * (bx1 - bx0)).mean())
            region = {"bbox": box, "col": col, "h_rel": round(h_rel, 2), "w_rel": round(w_rel, 2),
                      "density": round(density, 3), "status": "candidate", "why": None}
            if h_rel < MIN_H:
                region.update(status="rejected", why="short")
            elif h_rel > MAX_H:
                region.update(status="rejected", why="tall")
            elif w_rel < MIN_W:
                region.update(status="rejected", why="narrow")
            elif solid >= SOLID_SHARE:
                region.update(status="rejected", why="rule")
            elif density < DENSITY[0]:
                region.update(status="rejected", why="sparse")
            elif density > DENSITY[1]:
                region.update(status="rejected", why="dense")
            out.append(region)
    return out


def uncovered_regions(img, lines: list[dict], columns: list | None, typical_h: float, page_w: int, page_h: int,
                      rule_x: int | None = None, rule_y: int | None = None, stamps: list[dict] | None = None,
                      max_regions: int = MAX_REGIONS) -> dict:
    """
    {"residual_share", "regions", "capped"}: the ink no recognised line covers,
    as regions with a verdict each (band_regions), the candidates first and at
    most `max_regions` of them, the largest by ink.

    Excluded before anything is measured: the header and footer bands, the
    scanner's stamps, the page's edges, a halo around the vertical rule and
    around the footnote rule, and every recognised box padded by PAD.
    """
    import numpy as np
    ink = ink_mask(img)
    h, w = ink.shape
    total = int(ink.sum())
    pad = max(1, int(round(PAD * typical_h)))
    residual = ink & ~covered_mask(ink.shape, lines, pad)
    residual[:int(h * HEADER_BAND)] = False
    residual[int(h * (1 - FOOTER_BAND)):] = False
    edge = int(w * EDGE)
    residual[:, :edge] = False
    residual[:, w - edge:] = False
    for s in stamps or []:
        x0, y0, x1, y1 = s["bbox"]
        residual[max(0, int(y0) - pad):min(h, int(y1) + pad), max(0, int(x0) - pad):min(w, int(x1) + pad)] = False
    if rule_x is not None:
        halo = int(w * RULE_HALO)
        residual[:, max(0, rule_x - halo):min(w, rule_x + halo)] = False
    if rule_y is not None:
        halo = max(1, int(round(RULE_Y_HALO * typical_h)))
        residual[max(0, rule_y - halo):min(h, rule_y + halo)] = False
    share = float(residual.sum()) / max(total, 1)
    if share < MIN_RESIDUAL:
        return {"residual_share": round(share, 4), "regions": [], "capped": False}

    bands = [(int(lo), int(hi)) for lo, hi in columns] if columns else [(0, w)]
    regions: list[dict] = []
    for col, (lo, hi) in enumerate(bands):
        regions.extend(band_regions(residual, max(lo, edge), min(hi, w - edge), typical_h, col))
    # a region mostly inside a recognised box is that box, drawn a little short
    boxes = np.array([ln["bbox"] for ln in lines], dtype=float) if lines else np.zeros((0, 4))
    for r in regions:
        if r["status"] != "candidate":
            continue
        x0, y0, x1, y1 = r["bbox"]
        area = max(1, (x1 - x0) * (y1 - y0))
        if boxes.size:
            ix = np.clip(np.minimum(boxes[:, 2], x1) - np.maximum(boxes[:, 0], x0), 0, None)
            iy = np.clip(np.minimum(boxes[:, 3], y1) - np.maximum(boxes[:, 1], y0), 0, None)
            if float((ix * iy).max()) / area > LINE_OVERLAP_MAX:
                r.update(status="rejected", why="overlaps-line")
    candidates = [r for r in regions if r["status"] == "candidate"]
    capped = len(candidates) > max_regions
    if capped:
        by_ink = sorted(candidates, key=lambda r: -r["density"] * (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
        for r in by_ink[max_regions:]:
            r.update(status="rejected", why="capped")
    regions.sort(key=lambda r: (r["status"] != "candidate", r["bbox"][1], r["bbox"][0]))
    return {"residual_share": round(share, 4), "regions": regions, "capped": capped}


def crop_box(region: dict, band: tuple[int, int], typical_h: float, page_w: int, page_h: int) -> list[int]:
    """The region padded by half a line each way, kept inside its column band and the page."""
    x0, y0, x1, y1 = region["bbox"]
    pad = int(round(0.5 * typical_h))
    lo, hi = band
    return [max(lo, x0 - pad, 0), max(0, y0 - pad), min(hi, x1 + pad, page_w), min(page_h, y1 + pad)]


def is_text_reading(text: str, lang: str) -> str | None:
    """None when a reading looks like text; else why not: "empty", "no-letters" or "junk"."""
    import unicodedata
    t = (text or "").strip()
    if not t:
        return "empty"
    chars = [c for c in t if not c.isspace()]
    # letters and the marks that ride on them; a danda, a bar or a dot is neither
    letters = [c for c in chars if c.isalpha() or unicodedata.category(c).startswith("M")]
    if len(letters) < 0.5 * len(chars):
        return "no-letters"
    if len(set(chars)) <= 1 or len(t.split()) < 1:
        return "junk"
    return None
