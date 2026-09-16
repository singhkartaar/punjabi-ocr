"""
Which essay a PDF is, and which work it belongs to.

The filenames carry everything needed, in three shapes:

  1492003384127-Dhooja-Bhau-Part-2.pdf      a 10-digit upload stamp, then the
                                            3-digit essay number, then the title
  Lekh-110-Simran-Part-1-SS-English-_-Punjabi.pdf
  1563125358Lekh_8_Gurprasad_SS_JK.pdf      stamp, then "Lekh <n>"
  1491937205spritual_oasis.pdf              stamp, then a book title (no essay number)

The essay number is checked against the "L127.3" markers the pages carry, so a
filename that disagrees with its own content is reported rather than trusted.

Parts are stitched: "Sangat Part 1".."Sangat Part 14" is ONE work of 14 parts,
because a reader asking about sangat wants the essay, not a file. The work key
is the title with the part stripped, so adding a part later slots in.
"""
from __future__ import annotations
import os
import re

STAMP_ESSAY = re.compile(r"^([0-9]{10})([0-9]{3})-(.+)$")
STAMP_LEKH = re.compile(r"^([0-9]{10})Lekh[_-]([0-9]{1,4})[_-](.+)$", re.I)
LEKH = re.compile(r"^Lekh[_-]([0-9]{1,4})[_-](.+)$", re.I)
STAMP_ONLY = re.compile(r"^([0-9]{10})(.+)$")
PART = re.compile(r"[_-]Part[_-]([0-9]{1,3})(?:[_-]and[_-]([0-9]{1,3}))?", re.I)
# The bilingual tag the scans carry, stripped whole before tokenising.
NOISE = re.compile(r"[_-]?(?:SS[_-]English[_-]_[_-]Punjabi|English[_-]_[_-]Punjabi)$", re.I)

# Initials of the people who typed and checked each scan, and the editing-round
# words, left in the filenames. They are not part of any title.
NOISE_TOKENS = {"ss", "jk", "sv", "gd", "ssjk", "jkss", "jksv", "gdjkss", "ssjkss",
                "uni", "edit", "edit2", "2jk", "jk2", "fixed", "final", "rev"}

# The same essay is spelt two ways across its own parts. Without this,
# "Hukum Part 1" and "Hukam Part 2".."Part 8" are two works.
ALIASES = {"hukum": "hukam", "dharam-parchar": "dharam-parchaar",
           "dharan-ja-mazab": "dharam-ja-mazab"}

# The five books have typos in their filenames; they are the display titles a
# reader sees, so they are corrected here rather than shown as typed.
TITLES = {"hukam": "Hukam", "divine-wil": "Divine Will", "spritual-oasis": "Spiritual Oasis",
          "power-of-thougts-2": "Power of Thoughts, Part 2",
          "transformation-of-egoistic-conciousness": "Transformation of Egoistic Consciousness",
          "universal-religion": "Universal Religion"}


def _clean(title: str) -> str:
    prev = None
    while prev != title:
        prev = title
        title = NOISE.sub("", title)
    tokens = [t for t in re.split(r"[\s_-]+", title) if t]
    kept = [t for t in tokens if t.lower() not in NOISE_TOKENS]
    # a bare number left behind by an editing round ("Shabad edit 2 JK fixed")
    # is not a title; one that survived untouched ("Power of Thoughts 2") is
    if len(kept) < len(tokens):
        while len(kept) > 1 and kept[-1].isdigit():
            kept.pop()
    return " ".join(kept).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def parse_filename(path: str) -> dict:
    """
    @returns {"essay": int|None, "part": int|None, "title": str, "work": str,
              "work_title": str, "file": str}
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    essay, rest = None, stem
    m = STAMP_ESSAY.match(stem) or STAMP_LEKH.match(stem)
    if m:
        essay, rest = int(m.group(2)), m.group(3)
    elif LEKH.match(stem):
        m = LEKH.match(stem)
        essay, rest = int(m.group(1)), m.group(2)
    elif STAMP_ONLY.match(stem):
        rest = STAMP_ONLY.match(stem).group(2)

    part = None
    pm = PART.search(rest)
    if pm:
        part = int(pm.group(1))
        rest = rest[:pm.start()] + rest[pm.end():]
    work_title = _clean(rest)
    work = ALIASES.get(_slug(work_title), _slug(work_title))
    work_title = TITLES.get(work, work_title)
    title = work_title + (f", Part {part}" if part else "")
    # The five works loose in the root are Bau Ji's own English and may be
    # quoted as written. Everything in English/ and Punjabi/ is a translation of
    # his Punjabi, so an answer should say what he says rather than reproduce a
    # translator's wording as if it were his.
    folder = source_folder(path)
    return {"essay": essay, "part": part, "title": title, "work": work,
            "work_title": work_title, "file": os.path.basename(path),
            "folder": folder, "original": folder == "root"}


def source_folder(path: str) -> str:
    """Which of the three folders a PDF came from: "root", "English" or "Punjabi"."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent if parent in ("English", "Punjabi") else "root"


def list_sources(root: str) -> list[str]:
    """Every PDF under root and its immediate subfolders, in a stable order."""
    out = []
    for sub in ("", "English", "Punjabi"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(".pdf"):
                out.append(os.path.join(d, name))
    return out
