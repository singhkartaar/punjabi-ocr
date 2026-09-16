"""
A PDF typed in a legacy Gurmukhi font, read without OCR.

The Vaaran Bhai Gurdas file is not a scan: its text layer is real, but the
font is GurbaniAkharHeavy and the characters are the keystrokes of that
keyboard ("suxI pukwr dwqwr pRB gur nwnk"), which no Unicode-aware reader
recognises. Converting the keystrokes is a font map, not a reading, and the
map already exists in packages/search-core/src/gurmukhi.js (anvaad-js), so
this module extracts the text with PyMuPDF and pipes it through a small Node
helper rather than carrying a second copy of the table.

Lines are grouped into paragraphs by the vertical gap, with the "(1-23-1)"
vaar-pauri-line ids the file prints after each line kept as the paragraph's
marker, so a Vaar line can later be checked against BaniDB source B.
Returns read_pdf()'s shape.
"""
from __future__ import annotations
import os
import re
import statistics
import subprocess

from lib.paths import ROOT

CLI = os.path.join(ROOT, "pipeline", "node", "src", "lib", "legacy-font-cli.js")
LEGACY = re.compile(r"GurbaniAkhar|GurbaniLipi|AnmolLipi|Satluj|Amrit|Joy|Punjabi|Gurmukhi|Chatrik|Asees", re.I)
MARKER = re.compile(r"\(\s*(\d{1,2})\s*-\s*(\d{1,3})\s*-\s*(\d{1,3})\s*\)")
LEADING_BREAK = 1.6


def to_unicode(lines: list[str]) -> list[str]:
    """Legacy-font strings -> Unicode, via the Node helper, one call per document."""
    if not lines:
        return []
    if not os.path.exists(CLI):
        raise RuntimeError("the legacy Gurmukhi font converter (%s) is not in this checkout; it is a Node "
                           "helper of the private pipeline. Convert the PDF to Unicode another way and set "
                           "reader: pdf-text, or OCR it with reader: ocr." % os.path.relpath(CLI, ROOT))
    proc = subprocess.run(["node", CLI], input="\n".join(lines) + "\n", capture_output=True,
                          text=True, encoding="utf-8", check=True)
    out = proc.stdout.split("\n")
    out = out[:len(lines)] + [""] * max(0, len(lines) - len(out))
    return out


def read_legacy_pdf(path: str) -> dict:
    import pymupdf
    doc = pymupdf.open(path)
    pages_raw = []
    raw_lines: list[str] = []
    for n, page in enumerate(doc, start=1):
        d = page.get_text("dict")
        lines = []
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                fonts = {s.get("font", "") for s in spans}
                x0, y0, x1, y1 = line["bbox"]
                lines.append({"x0": x0, "y": -y0, "size": y1 - y0, "text": text,
                              "legacy": any(LEGACY.search(f) for f in fonts)})
                raw_lines.append(text if any(LEGACY.search(f) for f in fonts) else "")
        pages_raw.append(lines)
    doc.close()
    converted = to_unicode(raw_lines)
    k = 0
    pages = []
    sizes = []
    for n, lines in enumerate(pages_raw, start=1):
        paras = []
        cur = None
        gaps = [lines[i - 1]["y"] - lines[i]["y"] for i in range(1, len(lines))]
        normal = statistics.median(gaps) if gaps else 0.0
        prev_y = None
        marker = None
        for ln in lines:
            text = converted[k] if ln["legacy"] else ln["text"]
            k += 1
            m = MARKER.search(ln["text"])
            if m and marker is None:
                marker = "%s.%s" % (m.group(1), m.group(2))
            text = MARKER.sub("", text).strip()
            if not text:
                prev_y = ln["y"]
                continue
            gap = (prev_y - ln["y"]) if prev_y is not None else 0.0
            if cur is None or (normal and gap > normal * LEADING_BREAK):
                cur = {"text": text, "style": "quote", "italic": True, "x0": ln["x0"], "size": ln["size"]}
                paras.append(cur)
            else:
                cur["text"] += " " + text
            prev_y = ln["y"]
            sizes.append(ln["size"])
        pages.append({"page": n, "marker": marker, "spread": False, "columns": 0, "paragraphs": paras})
    return {"path": path, "body_size": statistics.median(sizes) if sizes else 10.0, "pages": pages}
