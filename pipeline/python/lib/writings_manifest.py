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
  kind          what the work is to the scripture: essay (about it, quoting it) |
                translation | word-meaning | commentary | reference; absent = essay.
                The reader of a paired page and the links it writes depend on it.
  translate     whether the work is rendered into English for the English corpus
                (search, Ask); absent = yes for a work not in English. A work kept
                in its own language is still embedded and served in that language.
  layout        how a page is read: auto | paired-columns | columns; absent = auto.
                columns is the reading order before pairing existed (left column,
                then right); paired-columns forces verse-and-explanation pairing.
  angs          [from, to]: the angs the work covers, for a page whose header
                gives no usable ang and for the works table; absent = unknown
  header_pattern  a regular expression with the named groups ang_from, ang_to,
                book_page, section, when the running header is not the Santhya's
                "<section> ( <page> ) (<bani>-ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ <ang>-<ang>"
  coverage      whether the merge reads the ink its lines left uncovered
                (22_ocr_merge.py --coverage); absent = the driver's default
  glossary      a file beside manifest.json with the book's terms (lib/mt_glossary):
                named in the translation prompt and checked in the English
  prompt        stock | rules | terms: the translator's instruction (26 --prompt)
  arbiter       none | self | llama | vertex: who retranslates a paragraph the
                checks refused (26 --arbiter); absent = none
  correct_agreed  whether the merge also corrects a word every engine misread
                alike, one seeded confusion from a prose word (22 --correct-agreed);
                for a grey scan whose subjoined ra both engines lose; absent = off
  corpus        the corpus the books join, when it is not the default data/writings:
                "barusahib" puts the paragraphs in data/barusahib/ beside the
                roster's works, and the indexes in barusahib-<lang> (27_ingest_book.py)
  title_en      the work's title in English, where `title` is the Gurmukhi one
"""
from __future__ import annotations
import json
import os
import re

from lib.writings_works import list_sources as _list_files, parse_filename

MANIFEST = "manifest.json"
DEFAULTS = {"language": "en", "reader": None, "licence": None, "original": True,
            "quote_policy": None, "author": None, "scripture": "G",
            "kind": "essay", "layout": "auto", "translate": None, "angs": None, "header_pattern": None,
            # style: how a notation book (kind "notation": 29-34, lib/notation*.py) lays its grids out
            # (lib/notation.DEFAULT_STYLE), a work's keys over the folder's
            "style": None}
KINDS = ("essay", "translation", "word-meaning", "commentary", "reference", "notation")
LAYOUTS = ("auto", "paired-columns", "columns")
PROMPTS = ("stock", "rules", "terms")             # 26_translate_writings.py --prompt
ARBITERS = ("none", "self", "llama", "vertex")    # 26_translate_writings.py --arbiter


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
        meta["translate"] = meta["language"] != "en"
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
    for k in ("language", "reader", "licence", "quote_policy", "author", "scripture", "died", "bleed", "coverage",
              "correct_agreed", "kind", "layout", "angs", "header_pattern", "glossary", "prompt", "arbiter",
              "corpus", "title_en"):
        if k in merged:
            meta[k] = merged[k]
    meta.setdefault("language", "en")
    meta.setdefault("scripture", "G")
    if meta.get("kind") not in KINDS:
        raise ValueError("%s: kind must be one of %s, not %r" % (name, ", ".join(KINDS), meta.get("kind")))
    if meta.get("layout") not in LAYOUTS:
        raise ValueError("%s: layout must be one of %s, not %r" % (name, ", ".join(LAYOUTS), meta.get("layout")))
    if meta.get("prompt") not in (None,) + PROMPTS:
        raise ValueError("%s: prompt must be one of %s, not %r" % (name, ", ".join(PROMPTS), meta.get("prompt")))
    if meta.get("arbiter") not in (None,) + ARBITERS:
        raise ValueError("%s: arbiter must be one of %s, not %r" % (name, ", ".join(ARBITERS), meta.get("arbiter")))
    # a work not in English is translated unless the manifest says otherwise
    meta["translate"] = bool(merged["translate"]) if merged.get("translate") is not None else meta["language"] != "en"
    if meta["kind"] == "notation":
        from lib.notation import merge_style
        meta["style"] = merge_style(manifest.get("style"), entry.get("style"))
    return meta
