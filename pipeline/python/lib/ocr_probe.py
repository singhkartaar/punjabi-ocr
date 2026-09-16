"""
What kind of PDF is this, before anything tries to read it.

Three shapes arrive under one extension, and each needs a different reader:

  text         a real Unicode text layer (the Bau Ji essays): lib/writings_pdf.py
  legacy-font  a text layer in a pre-Unicode Gurmukhi font (GurbaniAkhar,
               AnmolLipi, Satluj ...): the bytes are ASCII keystrokes, so
               "suxI pukwr" is ਸੁਣੀ ਪੁਕਾਰ, and the fix is a font map, not OCR
  image        no text at all, or only a "Page N of 530" stamp the scanner
               added: lib/ocr_pages.py renders and the engines read

Within "image", two things decide the rendering options:

  bilevel      1-bit CCITT / JBIG2 scans are already binarised at their native
               resolution (measured: 2560x3300 on the Santhya, ~300 dpi) and
               need no contrast work
  tiled        pages built from many small JPEG strips (426 px wide on the
               sikhbookclub scans) must be rasterised whole; extracting the
               images would give strips, not pages

A few pages spread through the file are sampled rather than all of them: a
530-page scan is uniform, and reading every image header of a 112 MB file to
learn that is a minute nobody needs.
"""
from __future__ import annotations
import re

LEGACY_FONTS = re.compile(r"GurbaniAkhar|GurbaniLipi|AnmolLipi|AnmolUni?bani|Satluj|Amrit|Joy|"
                          r"Punjabi|Gurmukhi|Chatrik|Asees|DrChatrik", re.I)
STAMP = re.compile(r"^\s*(Page\s+\d+(\s+of\s+\d+)?|www\.\S+|\d{1,4})\s*$", re.I | re.M)
GURMUKHI = re.compile("[਀-੿]")
SAMPLE = 6                      # pages examined
MIN_TEXT_CHARS = 120            # per page, after stamps are removed, to count as a text layer


def _sample_indices(n: int, k: int = SAMPLE) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({int(i * (n - 1) / (k - 1)) for i in range(k)})


def probe_pdf(path: str) -> dict:
    """
    @returns {"shape": "text"|"legacy-font"|"image", "pages", "bilevel", "tiled",
              "dpi", "fonts", "text_chars_per_page", "images_per_page", "image_dims"}
    """
    import pymupdf
    doc = pymupdf.open(path)
    n = doc.page_count
    fonts: set[str] = set()
    text_chars, image_counts, dims, bpcs = [], [], [], []
    gurmukhi = False
    for i in _sample_indices(n):
        page = doc[i]
        for f in page.get_fonts(full=True):
            fonts.add(f[3])
        text = STAMP.sub("", page.get_text())
        text_chars.append(len(text.strip()))
        gurmukhi = gurmukhi or bool(GURMUKHI.search(text))
        infos = page.get_image_info()
        image_counts.append(len(infos))
        for info in infos:
            dims.append((int(info.get("width", 0)), int(info.get("height", 0))))
            bpcs.append(int(info.get("bpc", 8)))
    doc.close()

    avg_chars = sum(text_chars) / max(len(text_chars), 1)
    per_page = sum(image_counts) / max(len(image_counts), 1)
    widest = max((w for w, _ in dims), default=0)
    tallest = max((h for _, h in dims), default=0)
    page_w_in = None
    try:
        doc = pymupdf.open(path)
        r = doc[0].rect
        page_w_in = (r.height if r.width > r.height and per_page <= 1 else r.width) / 72.0
        doc.close()
    except Exception:                                   # noqa: BLE001
        pass
    dpi = round(widest / page_w_in) if page_w_in and widest else None

    if avg_chars >= MIN_TEXT_CHARS and gurmukhi:
        shape = "text"
    elif avg_chars >= MIN_TEXT_CHARS and any(LEGACY_FONTS.search(f) for f in fonts):
        shape = "legacy-font"
    elif avg_chars >= MIN_TEXT_CHARS:
        shape = "text"
    else:
        shape = "image"
    return {
        "shape": shape, "pages": n,
        "bilevel": bool(bpcs) and max(bpcs) == 1,
        "tiled": per_page > 3 and tallest > 0 and widest > 0 and tallest < widest * 1.2,
        "dpi": dpi, "fonts": sorted(fonts),
        "text_chars_per_page": round(avg_chars), "images_per_page": round(per_page, 1),
        "image_dims": sorted(set(dims))[:5],
    }
