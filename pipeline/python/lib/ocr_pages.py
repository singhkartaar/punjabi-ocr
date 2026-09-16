"""
A scanned PDF as page images an OCR engine can read.

Pages are RENDERED, never extracted. Two of the sample books store each page as
eight JPEG strips 426 px wide, the Santhya stores one CCITT bitmap per page
with a /Rotate, and the Ten Masters scan is a single 770x1240 JPEG. Rendering
through PyMuPDF at a nominal 300 dpi makes all of them the same thing
downstream: one grayscale PNG per page in the page's reading orientation. An
extracted image would be a strip, a rotated bitmap, or a 100 dpi picture.

Three clean-ups, each measured on the samples and each off where it does not apply:

  crop_border   the Kabit scan has the page skewed inside a black scanner
                frame. Rows and columns that are almost solid ink at the edges
                are cut before deskewing, or the frame's edges win the angle.
  deskew        the angle that maximises the variance of the horizontal ink
                profile, searched over +-3 degrees. Coarse quarter-degree pass
                then a fine pass around the best; text lines are what make the
                profile spiky, so the best angle is the one that makes lines
                horizontal. Recorded per page so ground-truth boxes stay valid.
  suppress_bleed the 100 dpi Ten Masters scan shows the reverse side through
                the paper. Dividing by a heavily blurred copy flattens the paper
                tone, then everything lighter than the text is pushed to white.
                Off by default: on a 1-bit scan there is nothing to flatten.

The page's own text layer -- on these scans only the "Page 41 of 530" or
"www.sikhbookclub.com" stamp the scanner software added -- is read and kept
with its box, so lib/ocr_zones.py can drop that region by geometry rather than
only by matching the words an engine happened to read off it.
"""
from __future__ import annotations
import os

import numpy as np

DPI = 300
SKEW_RANGE = 3.0          # degrees searched either side of zero
SKEW_COARSE = 0.25
SKEW_FINE = 0.05
SKEW_WORK_H = 1200        # rows of the downscaled copy the search runs on
BORDER_INK = 0.5          # a row/col this dark at the edge is frame, not page
BORDER_MAX = 0.12         # never cut more than this share from any edge
BLEED_BLUR = 51           # median-blur kernel for the paper-tone estimate
BLEED_WHITE = 200         # after flattening, anything lighter than this is paper
MIN_TEXT_INK = 0.002      # a page with less ink than this is blank


def render(page, dpi: int = DPI) -> np.ndarray:
    """One page as an 8-bit grayscale array, in its reading orientation."""
    import pymupdf
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def _binary(img: np.ndarray) -> np.ndarray:
    """Ink as 1, paper as 0, by Otsu."""
    import cv2
    _, b = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def crop_border(img: np.ndarray):
    """
    (image, [top, bottom, left, right] rows/cols removed). Cuts inward from each
    edge while the edge is mostly ink, up to BORDER_MAX of the dimension.
    """
    b = _binary(img)
    h, w = b.shape
    rows, cols = b.mean(axis=1), b.mean(axis=0)
    lim_r, lim_c = int(h * BORDER_MAX), int(w * BORDER_MAX)
    top = 0
    while top < lim_r and rows[top] > BORDER_INK:
        top += 1
    bottom = 0
    while bottom < lim_r and rows[h - 1 - bottom] > BORDER_INK:
        bottom += 1
    left = 0
    while left < lim_c and cols[left] > BORDER_INK:
        left += 1
    right = 0
    while right < lim_c and cols[w - 1 - right] > BORDER_INK:
        right += 1
    # a frame is usually followed by a thin white gutter then the page edge
    # shadow; one more pass takes any residual dark band
    out = img[top:h - bottom, left:w - right]
    return out, [top, bottom, left, right]


def _rotate(img: np.ndarray, angle: float, fill: int = 255) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def _profile_score(binary_small: np.ndarray, angle: float) -> float:
    rot = _rotate(binary_small, angle, fill=0)
    prof = rot.sum(axis=1).astype(np.float64)
    return float(prof.var())


def deskew(img: np.ndarray):
    """(image rotated to horizontal text, angle applied in degrees)."""
    import cv2
    b = _binary(img)
    if b.mean() < MIN_TEXT_INK:
        return img, 0.0
    scale = SKEW_WORK_H / float(b.shape[0])
    small = cv2.resize(b, (max(1, int(b.shape[1] * scale)), SKEW_WORK_H), interpolation=cv2.INTER_AREA)
    coarse = np.arange(-SKEW_RANGE, SKEW_RANGE + 1e-9, SKEW_COARSE)
    best = max(coarse, key=lambda a: _profile_score(small, a))
    fine = np.arange(best - SKEW_COARSE, best + SKEW_COARSE + 1e-9, SKEW_FINE)
    best = float(max(fine, key=lambda a: _profile_score(small, a)))
    if abs(best) < SKEW_FINE:
        return img, 0.0
    return _rotate(img, best), round(best, 2)


def suppress_bleed(img: np.ndarray) -> np.ndarray:
    """Flatten the paper tone and whiten show-through; text ink is kept dark."""
    import cv2
    bg = cv2.medianBlur(img, BLEED_BLUR).astype(np.float32) + 1.0
    flat = np.clip(img.astype(np.float32) / bg * 255.0, 0, 255)
    flat[flat > BLEED_WHITE] = 255
    return flat.astype(np.uint8)


def page_text_layer(page, dpi: int = DPI) -> tuple[list[dict], list[dict]]:
    """
    (stamps, text_layer): what the PDF itself says is on this page, in pixels.

    On the Santhya that is one "Page 41 of 530" line the scanner added. On the
    Ten Masters scan it is a whole invisible OCR layer left by whoever scanned
    it -- a reading nobody asked for but a free second witness for the vote,
    served to the pipeline as the "pdftext" engine. The two are told apart
    line by line: a stamp matches the stamp patterns, everything else is text.
    """
    from lib.ocr_zones import is_stamp
    stamps, text = [], []
    scale = dpi / 72.0
    try:
        d = page.get_text("dict")
    except Exception:                                      # noqa: BLE001
        return stamps, text
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            s = "".join(sp.get("text", "") for sp in line.get("spans", [])).strip()
            if not s:
                continue
            x0, y0, x1, y1 = line["bbox"]
            rec = {"bbox": [int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)], "text": s}
            (stamps if is_stamp(s) else text).append(rec)
    return stamps, text


def page_stamps(page, dpi: int = DPI) -> list[dict]:
    return page_text_layer(page, dpi)[0]


def process(img: np.ndarray, do_crop: bool, do_deskew: bool, do_bleed: bool):
    """The clean-up chain on one rendered page. Returns (image, info)."""
    info: dict = {"cropped": None, "angle": 0.0, "bleed": bool(do_bleed)}
    if do_crop:
        img, info["cropped"] = crop_border(img)
    if do_bleed:
        img = suppress_bleed(img)
    if do_deskew:
        img, info["angle"] = deskew(img)
    return img, info


def render_pages(pdf: str, indices: list[int], out_dir: str, dpi: int = DPI,
                 do_crop: bool = True, do_deskew: bool = True, do_bleed: bool = False,
                 force: bool = False) -> list[dict]:
    """
    Render the given 0-based page indices to out_dir/NNNN.png.

    @returns [{"page", "file", "w", "h", "angle", "cropped", "stamps", "skipped"}]
    Runs in a worker process; opens the PDF itself so nothing unpicklable crosses.
    """
    import cv2
    import pymupdf
    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(pdf)
    out = []
    for i in indices:
        n = i + 1
        path = os.path.join(out_dir, "%04d.png" % n)
        rec = {"page": n, "file": os.path.basename(path)}
        if os.path.exists(path) and not force:
            rec["skipped"] = True
            out.append(rec)
            continue
        page = doc[i]
        img = render(page, dpi)
        img, info = process(img, do_crop, do_deskew, do_bleed)
        cv2.imwrite(path, img)
        stamps, text_layer = page_text_layer(page, dpi)
        rec.update({"w": int(img.shape[1]), "h": int(img.shape[0]), "stamps": stamps,
                    "text_layer": text_layer, **info})
        out.append(rec)
    doc.close()
    return out


def parse_pages(spec: str | None, total: int) -> list[int]:
    """'1-10,15' -> 0-based indices; None -> every page."""
    if not spec:
        return list(range(total))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
        else:
            lo = hi = int(part)
        for n in range(max(1, lo), min(total, hi) + 1):
            out.add(n - 1)
    return sorted(out)
