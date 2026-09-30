"""
Image primitives for notation grids: rules, bars, tables, bands, components, crops.

A notation page is line art -- a 1-bit scan of a table or of rows separated
by short vertical bars -- and the marks that carry meaning (komal underlines,
octave dots, grouping arcs, tivra ticks) are shapes no OCR engine reads. So
the grid parser works from the page's own ink: it finds the long horizontal
and vertical rules with morphology, the short bars that divide vibhags, the
bands of ink that are rows, and the small connected components around each
glyph. Everything here takes and returns pixel coordinates of the page PNG
20_ocr_pages.py wrote (300 dpi, deskewed), the same frame the OCR boxes are
in, and nothing here decides what a shape means -- lib/notation_grid.py does.

cv2 and numpy are imported inside the functions, as lib/ocr_pages.py does,
so the text-only modules import without them.
"""
from __future__ import annotations
import os


def binarise(img):
    """Ink as 255 on 0, by Otsu (the same threshold lib/ocr_layout.stroke_width uses)."""
    import cv2
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def horizontal_rules(ink, min_len: int, merge_px: int = 3) -> list[dict]:
    """
    Long horizontal strokes: [{"y", "x0", "x1", "thick"}] sorted by y. A rule
    is at least `min_len` px of unbroken ink; rules within `merge_px` rows of
    one another (a thick rule read twice) are merged.
    """
    import cv2
    import numpy as np
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, int(min_len)), 1))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    rules = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if w >= min_len:
            rules.append({"y": int(y + h / 2), "x0": int(x), "x1": int(x + w), "thick": int(h)})
    rules.sort(key=lambda r: r["y"])
    merged: list[dict] = []
    for r in rules:
        if merged and abs(r["y"] - merged[-1]["y"]) <= merge_px:
            m = merged[-1]
            m["x0"], m["x1"] = min(m["x0"], r["x0"]), max(m["x1"], r["x1"])
            m["thick"] = max(m["thick"], r["thick"])
        else:
            merged.append(dict(r))
    return merged


def vertical_rules(ink, min_len: int, merge_px: int = 3) -> list[dict]:
    """Long vertical strokes: [{"x", "y0", "y1", "thick"}] sorted by x."""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, int(min_len))))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    rules = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if h >= min_len:
            rules.append({"x": int(x + w / 2), "y0": int(y), "y1": int(y + h), "thick": int(w)})
    rules.sort(key=lambda r: r["x"])
    merged: list[dict] = []
    for r in rules:
        if merged and abs(r["x"] - merged[-1]["x"]) <= merge_px:
            m = merged[-1]
            m["y0"], m["y1"] = min(m["y0"], r["y0"]), max(m["y1"], r["y1"])
            m["thick"] = max(m["thick"], r["thick"])
        else:
            merged.append(dict(r))
    return merged


def bars(ink, band: tuple[int, int], min_h: int, max_w: int = 8, x0: int = 0, x1: int | None = None) -> list[int]:
    """
    The x centres of short vertical strokes inside one row band (y0, y1):
    the bars a bar-only grid draws between vibhags. A bar is at least
    `min_h` tall and at most `max_w` wide; anything wider is a glyph.
    """
    import cv2
    y0, y1 = band
    x1 = ink.shape[1] if x1 is None else x1
    strip = ink[y0:y1, x0:x1]
    if strip.size == 0:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, int(min_h))))
    opened = cv2.morphologyEx(strip, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if h >= min_h and w <= max_w:
            out.append(int(x0 + x + w / 2))
    return sorted(out)


def tables(h_rules: list[dict], v_rules: list[dict], tol: int = 6) -> list[dict]:
    """
    Ruled tables: groups of at least three horizontal rules of similar
    extent crossed by at least two vertical rules that span them.
    [{"bbox": [x0, y0, x1, y1], "rows": [y...], "cols": [x...]}]
    """
    out = []
    used = set()
    for i, top in enumerate(h_rules):
        if i in used:
            continue
        group = [top]
        for j in range(i + 1, len(h_rules)):
            r = h_rules[j]
            overlap = min(top["x1"], r["x1"]) - max(top["x0"], r["x0"])
            if overlap >= 0.6 * min(top["x1"] - top["x0"], r["x1"] - r["x0"]):
                group.append(r)
        if len(group) < 3:
            continue
        y0, y1 = group[0]["y"], group[-1]["y"]
        gx0, gx1 = min(r["x0"] for r in group), max(r["x1"] for r in group)
        cols = [v["x"] for v in v_rules if v["y0"] <= y0 + tol and v["y1"] >= y1 - tol and gx0 - tol <= v["x"] <= gx1 + tol]
        if len(cols) < 2:
            continue
        used.update(h_rules.index(r) for r in group)
        out.append({"bbox": [int(min(gx0, cols[0])), int(y0), int(max(gx1, cols[-1])), int(y1)],
                    "rows": [int(r["y"]) for r in group], "cols": [int(c) for c in cols]})
    return out


def row_bands(ink, bbox: list[int], min_gap: int, min_ink: int = 2) -> list[tuple[int, int]]:
    """
    Horizontal bands of ink inside bbox, as (y0, y1) in page coordinates:
    rows of text and of swaras. Gaps shorter than `min_gap` do not split a
    band (the dots above and below a swara row belong to it).
    """
    import numpy as np
    x0, y0, x1, y1 = bbox
    region = ink[y0:y1, x0:x1]
    if region.size == 0:
        return []
    profile = (region > 0).sum(axis=1)
    bands: list[list[int]] = []
    in_band, start, gap = False, 0, 0
    for i, v in enumerate(profile):
        if v >= min_ink:
            if not in_band:
                if bands and i - bands[-1][1] < min_gap:
                    in_band, start = True, bands.pop()[0]
                else:
                    in_band, start = True, i
            gap = 0
        elif in_band:
            gap += 1
            if gap >= 1:
                bands.append([start, i - gap + 1])
                in_band = False
    if in_band:
        bands.append([start, len(profile)])
    return [(y0 + a, y0 + b) for a, b in bands if b > a]


def components(ink, bbox: list[int] | None = None, min_area: int = 2) -> list[dict]:
    """Connected components, page-absolute: [{"x0","y0","x1","y1","w","h","area","cx","cy"}]."""
    import cv2
    if bbox is None:
        x0, y0 = 0, 0
        region = ink
    else:
        x0, y0, x1, y1 = bbox
        region = ink[y0:y1, x0:x1]
    if region.size == 0:
        return []
    n, _, stats, cents = cv2.connectedComponentsWithStats(region, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append({"x0": int(x0 + x), "y0": int(y0 + y), "x1": int(x0 + x + w), "y1": int(y0 + y + h),
                    "w": int(w), "h": int(h), "area": int(area), "cx": float(x0 + cents[i][0]), "cy": float(y0 + cents[i][1])})
    return out


def erase(ink, h_rules: list[dict] | None = None, v_rules: list[dict] | None = None, bars_xy: list[tuple[int, int, int]] | None = None, pad: int = 1):
    """A copy of the ink with the given rules and bars whitened, so glyph analysis does not see them."""
    out = ink.copy()
    H, W = out.shape[:2]
    for r in h_rules or []:
        y0, y1 = max(0, r["y"] - r["thick"] // 2 - pad), min(H, r["y"] + r["thick"] // 2 + pad + 1)
        out[y0:y1, max(0, r["x0"] - pad):min(W, r["x1"] + pad)] = 0
    for r in v_rules or []:
        x0, x1 = max(0, r["x"] - r["thick"] // 2 - pad), min(W, r["x"] + r["thick"] // 2 + pad + 1)
        out[max(0, r["y0"] - pad):min(H, r["y1"] + pad), x0:x1] = 0
    for x, y0, y1 in bars_xy or []:
        out[max(0, y0 - pad):min(H, y1 + pad), max(0, x - 4 - pad):min(W, x + 4 + pad)] = 0
    return out


def diagonal_ink(ink, bbox: list[int] | None = None) -> float:
    """
    The share of long straight segments in the region that run diagonally
    (20-70 degrees): a printed diagonal watermark over a grid shows here,
    text and rules do not.
    """
    import cv2
    import numpy as np
    region = ink if bbox is None else ink[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if region.size == 0:
        return 0.0
    min_len = max(40, int(0.1 * max(region.shape)))
    segs = cv2.HoughLinesP(region, 1, np.pi / 180, threshold=80, minLineLength=min_len, maxLineGap=6)
    if segs is None or len(segs) == 0:
        return 0.0
    diag = 0
    for (x0, y0, x1, y1) in segs[:, 0, :]:
        ang = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0))) % 180
        if 20 <= ang <= 70 or 110 <= ang <= 160:
            diag += 1
    return diag / len(segs)


def crop_bilevel(img, bbox: list[int], pad: int, dst: str) -> tuple[int, int]:
    """
    The region as a 1-bit PNG (the scans are bilevel; nothing is lost and the
    file is small), padded by `pad` px and clamped to the page. Returns (w, h).
    """
    import cv2
    from PIL import Image
    H, W = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    region = img[y0:y1, x0:x1]
    if region.ndim == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    Image.fromarray(bw).convert("1").save(dst, optimize=True)
    return int(x1 - x0), int(y1 - y0)


def thumbnail(img, bbox: list[int], dst: str, width: int = 320) -> tuple[int, int]:
    """A small grayscale PNG of the region for list views. Returns (w, h)."""
    import cv2
    from PIL import Image
    x0, y0, x1, y1 = bbox
    region = img[max(0, y0):y1, max(0, x0):x1]
    if region.ndim == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h = max(1, int(round(region.shape[0] * width / max(1, region.shape[1]))))
    small = cv2.resize(region, (width, h), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    Image.fromarray(small).save(dst, optimize=True)
    return width, h
