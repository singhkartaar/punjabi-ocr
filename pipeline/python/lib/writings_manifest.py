"""
Which author, language and licence a source folder carries.

The Bau Ji essays encode everything in their filenames and folder names, and
lib/writings_works.py reads them. Scanned books do not: the author, whether
the text is public domain, whether it may be quoted verbatim, and which reader
applies are facts about the book that no filename states. They live in a
manifest.json beside the PDFs, which never enters the repository.

A folder WITHOUT a manifest is read exactly as before -- parse_filename() and
the root/English/Punjabi convention -- so the essays are untouched.

Manifest shape: top-level keys are defaults, each work may override any of them.

  {"author": "...", "language": "pa", "reader": "ocr", "licence": "public-domain",
   "original": true, "quote_policy": "verbatim",
   "works": [{"file": "x.pdf", "work": "santhya", "part": 1, "title": "...", "book": "santhya-vol-1"}]}

  reader        ocr | legacy-font | pdf-text; absent = decided by lib/ocr_probe.py
  quote_policy  verbatim | summarise; absent = verbatim when licence is public-domain, else summarise
  book          the directory name under data/ocr/; absent = slug of the file stem
  scripture     BaniDB source code of the verse this book quotes (G SGGS, D Dasam
                Granth, B Vaaran, K Kabit) when it is not the Guru Granth Sahib
"""
from __future__ import annotations
import json
import os
import re

from lib.writings_works import list_sources as _list_files, parse_filename

MANIFEST = "manifest.json"
DEFAULTS = {"language": "en", "reader": None, "licence": None, "original": True,
            "quote_policy": None, "author": None, "scripture": "G"}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_manifest(src: str) -> dict | None:
    """The manifest in a source folder, or None when the folder has none."""
    path = os.path.join(src, MANIFEST)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    if not isinstance(m.get("works"), list):
        raise ValueError("%s: 'works' must be a list" % path)
    return m


def list_sources(src: str, manifest: dict | None = None) -> list[str]:
    """PDFs to read, in manifest order when there is one; else the folder convention."""
    if manifest is None:
        return _list_files(src)
    out = []
    for w in manifest["works"]:
        p = os.path.join(src, w["file"])
        if os.path.isfile(p):
            out.append(p)
    return out


def parse_source(path: str, manifest: dict | None = None, roster: dict | None = None) -> dict:
    """
    parse_filename()'s keys plus language, licence, reader, book, quote_policy,
    author, scripture and any per-work extras (e.g. "bleed").

    @param roster  an author's listing kept in the repository (rosters/*.json),
        the other way a file gets named: a folder with no manifest is read
        through it, exactly as parse_filename() would alone.
    """
    if manifest is None:
        meta = parse_filename(path, roster)
        meta.update({k: v for k, v in DEFAULTS.items() if k not in meta})
        meta["quote_policy"] = "verbatim" if meta["original"] else "summarise"
        meta["book"] = _slug(os.path.splitext(os.path.basename(path))[0])
        meta["reader"] = "pdf-text"
        return meta
    name = os.path.basename(path)
    entry = next((w for w in manifest["works"] if w.get("file") == name), None)
    if entry is None:
        raise KeyError("%s is not listed in the manifest" % name)
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in manifest.items() if k != "works"})
    merged.update(entry)
    stem = os.path.splitext(name)[0]
    work_title = merged.get("title") or stem
    part = merged.get("part")
    if merged.get("quote_policy") is None:
        merged["quote_policy"] = "verbatim" if merged.get("licence") == "public-domain" else "summarise"
    meta = {
        "essay": None, "part": part, "title": work_title,
        "work": merged.get("work") or _slug(work_title),
        "work_title": re.sub(r",?\s*Part \d+$", "", work_title) if part else work_title,
        "file": name, "folder": "root", "original": bool(merged.get("original", True)),
        "book": merged.get("book") or _slug(stem),
    }
    for k in ("language", "reader", "licence", "quote_policy", "author", "scripture", "died", "bleed"):
        if k in merged:
            meta[k] = merged[k]
    meta.setdefault("language", "en")
    meta.setdefault("scripture", "G")
    return meta
