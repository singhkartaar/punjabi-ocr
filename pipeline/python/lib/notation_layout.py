"""
A notation page, cut into regions; a book's pages, joined into notations.

What a page of a keertan book holds, top to bottom: a heading (the number,
the raag used, the taal, the laya), the shabad's text with a reference to
where it stands in the Granth, then one grid per section -- ਅਸਥਾਈ, ਅੰਤਰਾ --
each a block of row pairs (swaras over syllables) divided by bars, sometimes
a marker row above it, and often a note to the singer at the end (ਬਾਕੀ
ਤੁਕਾਂ ਅੰਤਰੇ ਤੇ ਲਾਓ). A grid runs onto the next page when the page ends, and
a book that prints the shabad AFTER its grids exists (Sangeet Sagar), so the
page is only half the story: link_pages() strings the regions of a book's
pages into notation spans.

The cut is made from the merged OCR lines (22_ocr_merge.py) and the page's
own ink. The words decide the roles -- a heading names a raag or a taal, a
reference names an ang, a section label is one of a few words, a grid row
is mostly swaras, dashes and held marks -- and the ink decides the extents,
because Tesseract's lines over a grid are unreliable while the rows of ink
are not. Nothing here reads a cell; lib/notation_grid.py does that.
"""
from __future__ import annotations
import re

from lib.notation_text import is_heading_like, is_note_like, parse_heading, parse_ref, token_class
from lib.notation_vocab import gurmukhi_digits, marker_kind, section_label

GRID_CLASSES = ("swara", "held", "rest", "marker", "bar", "matra")
_BAR_CHARS = re.compile(r"[|¦│\[\]{}!]|(?<![।])।(?![।])")
_BAR_SPLIT = re.compile(r"[\s|¦│\[\]{}!]+|(?<![।])।(?![।])")
_PAGENO = re.compile(r"^[\s।॥()\-]*[0-9੦-੯]{1,4}[\s।॥()\-]*$")
_COUNT_LINE = re.compile(r"[\s0-9੦-੯॥।]+")
_NUMBER_LINE = re.compile(r"^\s*\(?\s*[0-9੦-੯]{1,4}\s*[.)।]?\s*$")                  # "੧੮.", "(੨)"
_GRANTH_HEADING = re.compile(r"(?:ਮਹਲਾ|ਮਹੱਲਾ|ਮਃ|ਮ[:ਃ])\s*[੧-੯1-9]")
_MARKER_GLYPH = re.compile(r"^[x×X+०੦0oO°\[\)\(\]ਨਤੇੜੇ੨੩੪੧2345]{1,3}$")


# ---- lines -----------------------------------------------------------------

# The words a book uses when it describes a raag before its shabads -- the
# vaadi and samvaadi, the thaat, the pakad, aaroh and avroh, the swar-vistaar,
# a "ਰਾਗ ਪਰੀਚੈ" (introduction). Swar sequences set with dashes ("ਆਰੋਹ--ਸਗ,ਮਪਨਸੰ")
# read as grid rows by token shape, which is how a description under a raag's
# heading rode along with the notation before it (Gurbani Sangeet 1, p. 165,
# the second cut, 2 October 2026). A line that says one of these words is prose.
_DESCRIPTION_WORDS = re.compile(r"ਸੰਵਾਦੀ|ਸੌਵਾਦੀ|ਵਾਦੀ|ਠਾਠ|ਥਾਟ|ਪਕੜ|ਆਰ[ੋੌ]ਹ|ਅਰੋਹ|ਅਵਰ[ੋੌ]ਹ|ਵਿਸਥਾਰ|ਵਿਸਤਾਰ|ਪਰੀਚੈ|ਪਰਿਚੈ|ਪਰਿਚਯ"
                                r"|ਵਰਜਿਤ|ਨਿਆਸ|ਪ੍ਰਕ੍ਰਿਤੀ|ਮੁੱਖ ਅੰਗ|ਜਾਤੀ|ਜਾਤ\s*-|ਸਮਾਂ\s*-")
_RAAG_WORD = re.compile(r"^\W*ਰਾਗ[ੁ]?\b")
# A book that explains each shabad before its notation (Prof Paramjyot Singh's Swar Samund: the shabad and
# its reference, "ਭਾਵ-ਅਰਥ" and the meanings, "ਹੋਰ ਸ਼ਬਦ (ਅੰਮ੍ਰਿਤ ਕੀਰਤਨ) –" and the first lines of two more
# shabads for the same tune, each with its ang; the notation on the facing page). From either label to the
# notation's heading everything is prose: a sentence of the meanings is no grid row, and the other shabads'
# lines, which the corpus knows and which carry an ang, are not the notation's shabad (third cut, 5 October 2026).
_OTHERS_LABEL = re.compile(r"^\W*ਹੋਰ\s+ਸ਼ਬਦ\b")
_PROSE_LABEL = re.compile(r"^\W*(?:ਭਾਵ\s*[-–—]?\s*ਅਰਥ\W*$|ਪਦ\s*[-–—]?\s*ਅਰਥ\b|ਹੋਰ\s+ਸ਼ਬਦ\b)")      # ਪਦ ਅਰਥ: the word meanings under a pauri (Guru Angad Dev Sangeet Darpan)


def _describes_raag(region: dict) -> bool:
    """A heading that is the description's own: 'ਰਾਗੁ ਪਰੀਚੈ:-', 'ਥਾਟ - ਭੈਰਵ।'."""
    return bool(_DESCRIPTION_WORDS.search(region.get("text") or ""))


def _description_words(regions: list[dict], span: int = 4) -> set[str]:
    """
    The description's words in the prose at the head of `regions`: the next
    few text regions, stepping over grid and marker regions (a swar sequence
    set with dashes still reads as a grid row), stopping at a shabad, a
    reference, a section label or a break.
    """
    words: set[str] = set()
    seen = 0
    for r in regions:
        role = r.get("role")
        if role in ("grid", "marker"):
            continue
        if role not in ("text", "note") and not (role == "heading" and _DESCRIPTION_WORDS.search(r.get("text") or "")):
            break
        words.update(m.group(0).strip() for m in _DESCRIPTION_WORDS.finditer(r.get("text") or ""))
        seen += 1
        if seen >= span:
            break
    return words


def _description_follows(items: list[dict], i: int) -> bool:
    """Two or more of the description's words in the prose right after item i."""
    return len(_description_words([{**(it.get("region") or {}), "role": it["role"]} for it in items[i + 1:i + 8]])) >= 2


def classify_line(line: dict) -> str:
    """
    stamp | pageno | heading | section | ref | note | gurbani | grid | marker | text
    for one merged line. `kind` and `bold` come from the merge; the text
    decides the rest.
    """
    zone = line.get("zone")
    if zone in ("stamp", "pageno") or line.get("dropped"):
        return "stamp" if zone == "stamp" else "pageno"
    text = (line.get("text") or "").strip()
    if not text:
        return "text"
    # Tesseract glues a bar to its neighbour ("ਪ|ਪ", "ਨੁ।ਧੁ"): bars are separators here
    toks = [t for t in _BAR_SPLIT.split(text) if t]
    bar_count = len(_BAR_CHARS.findall(text))
    if _PAGENO.match(text) and len(toks) <= 2:
        return "pageno"
    if _COUNT_LINE.fullmatch(text) and "॥" in text:
        return "gurbani"                      # "੨ ॥ ੬ ॥": the verse's closing count, set on its own line
    if section_label(text) and len(toks) <= 3:
        return "section"
    if is_note_like(text):
        return "note"
    ref = parse_ref(text)
    if ref and (ref["ang_from"] or ref["source"] != "G") and len(toks) <= 12 \
            and (text.startswith("(") or text.startswith("[") or ref["span"][0] <= 2 or ref["source"] != "G"):
        return "ref"                          # "ਕਬਿੱਤ ਸਵੱਯੇ (ਭਾਈ ਗੁਰਦਾਸ ਜੀ)", "ਮੁਖ ਵਾਕ ਪਾਤਸ਼ਾਹੀ ੧੦ (ਦਸਮ ਗ੍ਰੰਥ 'ਚੋਂ)": the source line over the verse
    if _GRANTH_HEADING.search(text) and len(toks) <= 8 and "ਤਾਲ" not in text and not text.startswith("("):
        return "gurbani"                      # ਜੈਤਸਰੀ ਮਹਲਾ ੫ ਘਰੁ ੨: the Granth's own heading, the shabad's first line
    if is_heading_like(text):
        return "heading"
    if _DESCRIPTION_WORDS.search(text) and bar_count <= 2:
        return "text"                         # "ਆਰੋਹ--ਸਗ,ਮਪਨਸੰ।": a raag's description, not a grid row
    classes = [token_class(t) for t in toks]
    gridish = sum(c in GRID_CLASSES for c in classes) + bar_count
    if toks and all(_MARKER_GLYPH.match(t) or token_class(t) in ("marker", "matra", "bar") for t in toks) and len(toks) <= 20 and gridish + sum(1 for t in toks if _MARKER_GLYPH.match(t)) >= len(toks):
        return "marker"
    held = sum(1 for c in classes if c == "held") + bar_count
    total = len(toks) + bar_count
    if line.get("kind") in ("gurbani", "gurbani-unmatched") or "॥" in text or "।।" in text:
        # a verse -- unless it is a bol row that OCR happened to read as one
        if gridish / max(1, total) < 0.5 and held < 2:
            return "gurbani"
    if toks and gridish / total >= 0.5:
        return "grid"
    # a bol row: short syllables between held marks and bars; a verse of
    # short words has neither
    short = sum(1 for t in toks if len(t.strip("().,;")) <= 3)
    if len(toks) >= 4 and (short + gridish) / total >= 0.7 and held >= 2:
        return "grid"
    # bold alone says nothing: swaras print heavy too. A bold line with a
    # danda is a verse (a heading of the shabad, ਗਉੜੀ ਮਹਲਾ ੫ ॥)
    if line.get("bold") and len(toks) >= 3 and "।" in text:
        return "gurbani"
    return "text"


def _bbox_union(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ---- the page --------------------------------------------------------------

def page_layout(merged_lines: list[dict], page_w: int, page_h: int, style: dict, page: int,
                ink=None, body_h: int | None = None) -> dict:
    """
    {"page", "kind": notation|prose|blank, "regions": [Region, ...]} in
    reading order, a Region being {"role": heading|shabad|ref|section|
    marker|grid|note|text, "bbox", "lines": [merged line numbers], "text",
    "parsed": {...} for a heading/ref/section}. A grid region's bbox is the
    ink's extent between the lines that bound it when `ink` is given.
    """
    lines = [l for l in merged_lines if l.get("bbox")]
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    roles = [(l, classify_line(l)) for l in lines]
    body_h = body_h or _median([l["bbox"][3] - l["bbox"][1] for l, r in roles if r in ("text", "gurbani", "heading")] or [40])

    regions: list[dict] = []
    cur: dict | None = None

    def close():
        nonlocal cur
        if cur:
            regions.append(cur)
            cur = None

    # the running header the manifest names (style.running_header), when it is itself a line of Gurbani
    # ("ਭਗਤਿ ਹੇਤ ਗਾਵੈ ਰਵਿਦਾਸਾ", the title of Prof Tara Singh's book, over every left-hand page): the corpus
    # matches it, so by its words it is a verse, and it opened a shabad or joined the one under it
    named = (style or {}).get("running_header") or []
    named = [named] if isinstance(named, str) else list(named)
    prose = False
    for line, role in roles:
        if named and role == "gurbani" and page_h and line["bbox"][1] < 0.1 * page_h \
                and _header_line(line.get("text") or "", named):
            continue
        if role == "pageno" and line.get("zone") == "body" and page_h and 0.08 * page_h < line["bbox"][1] < 0.5 * page_h \
                and _BRACKET_NUMBER.match((line.get("text") or "").strip()):
            # "(੨)" at the left margin under the raag's heading: the notation's number, not a page's
            # (Guru Nanak Sangeet Padhti Granth sets the number at the left margin and the taal at the right)
            role = "heading"
        if role in ("stamp", "pageno"):
            continue
        n, bbox, text = line.get("n"), list(line["bbox"]), (line.get("text") or "").strip()
        if role in ("grid", "marker", "text") and not _WORD_CHAR.search(text) \
                and bbox[3] - bbox[1] < 0.5 * body_h and bbox[2] - bbox[0] < body_h:
            continue                                  # a speck read as "=" or "'" on a line of its own: a grid row of one mark it is not
        if _PROSE_LABEL.match(text):
            prose = "others" if _OTHERS_LABEL.match(text) else "meanings"
            close()
        elif prose and (role == "section" or (role == "heading" and parse_heading(text).get("taal"))):
            prose = False                             # the notation's own heading, or a section label: the prose is over
        if prose:
            # the meanings, the other shabads: one region of text, whatever its lines look like
            if cur and cur.get("prose"):
                cur["lines"].append(n); cur["bbox"] = _bbox_union([cur["bbox"], bbox]); cur["text"] += "\n" + text
                if prose == "others":
                    # a line of the list: another shabad for the tune, with what the corpus made of it (the record's `also`)
                    cur["others"].append({"text": text, "matches": [{k: m.get(k) for k in ("shabad_id", "line_id", "score", "source")}
                                                                    for m in (line.get("matches") or [])[:3]]})
            else:
                close()
                cur = {"role": "text", "bbox": bbox, "lines": [n], "text": text, "kinds": [], "prose": True}
                if prose == "others":
                    cur["others"] = []
            continue
        if role in ("heading", "section", "ref", "note", "marker"):
            close()
            region = {"role": role, "bbox": bbox, "lines": [n], "text": text}
            if role == "heading":
                region["parsed"] = parse_heading(text)
            elif role == "section":
                sec = section_label(text)
                region["parsed"] = {"kind": sec[0], "n": sec[1]} if sec else None
            elif role == "ref":
                region["parsed"] = parse_ref(text)
            elif role == "marker":
                region["parsed"] = {"marks": _marker_row(text)}
            regions.append(region)
            continue
        if role in ("gurbani", "text"):
            # a verse block; a text line inside one continues it, a text line
            # alone (an alaap word, an OCR shred) stays text
            if cur and cur.get("prose"):
                close()
            if cur and cur["role"] == "shabad":
                cur["lines"].append(n); cur["bbox"] = _bbox_union([cur["bbox"], bbox]); cur["text"] += "\n" + text
                cur["kinds"].append(line.get("kind")); cur["merged"].append(line)
            elif cur and cur["role"] == "grid" and role == "text" and len(text.split()) <= 3:
                # a fragment Tesseract cut out of a grid row
                cur["lines"].append(n); cur["bbox"] = _bbox_union([cur["bbox"], bbox])
            else:
                close()
                cur = {"role": "shabad" if role == "gurbani" or _looks_like_verse(text) else "text",
                       "bbox": bbox, "lines": [n], "text": text, "kinds": [line.get("kind")], "merged": [line]}
            continue
        if role == "grid":
            if cur and cur["role"] == "grid":
                cur["lines"].append(n); cur["bbox"] = _bbox_union([cur["bbox"], bbox])
            elif cur and cur["role"] == "text" and len(cur["lines"]) <= 1 and not _DESCRIPTION_WORDS.search(cur["text"]):
                # a stray fragment before the grid belongs to it (not a description's line: "ਵਾਦੀ--ਗ ਠਾਠ--ਖਮਾਜ")
                cur = {"role": "grid", "bbox": _bbox_union([cur["bbox"], bbox]), "lines": cur["lines"] + [n], "text": ""}
            else:
                close()
                cur = {"role": "grid", "bbox": bbox, "lines": [n], "text": ""}
            continue
    close()

    # the number and the taal of a notation set on one line, apart ("(੨)" at the left margin, "ਤੀਨ ਤਾਲ"
    # at the right): two headings on one baseline are one
    regions = _join_number_and_taal(regions, body_h)
    # a verse line Tesseract read as a bol row stands as a grid of a line or two right over the
    # verse block it opens: a line of words, or one the corpus matched to the block's shabad, is the block's
    regions = _fold_misread_verse(regions, {l.get("n"): l for l in lines}, body_h)
    # between a heading and the first section label, marker row or reference
    # lies the shabad's text, whatever the OCR made of its lines: a Kabitt
    # of Bhai Gurdas matches no corpus line and reads as prose
    regions = _shabad_between(regions, body_h)
    regions = _fold_shreds(regions)
    # a shabad block is a verse the corpus knows -- that is the link the
    # notation needs -- or a verse the book set over a printed reference (a
    # Kabitt of Bhai Gurdas, Dasam Bani, which the corpus does not hold).
    # Prose that mentions a line, tabla bols under a taal heading, and OCR
    # noise that happens to hold a danda are text.
    ref_before = (style or {}).get("ref_position") == "before"
    for i, r in enumerate(regions):
        if r["role"] != "shabad":
            continue
        # the printed reference stands after the verse, or (Gurmat Sangeet Darpan) above it
        beside = regions[i - 1] if ref_before and i > 0 else regions[i + 1] if not ref_before and i + 1 < len(regions) else None
        over_ref = bool(beside and beside["role"] == "ref" and (_verse_shaped(r["text"]) or _granth_set(r["text"], beside)))
        if not (_verse_block(r) or over_ref):
            r["role"] = "text"
    # a line naming another scripture (the Kabit Savaiye, the Dasam Bani) stands over its verse in every book,
    # and that verse matches no corpus line: the block under it is a shabad of that source, linked when the
    # scriptures store exists (docs/plans/scriptures-2026-10-02.md), a boundary now
    for i, r in enumerate(regions[:-1]):
        nxt = regions[i + 1]
        # "(ਸੁਪੁੱਤ੍ਰ ਗੁਰ ਅੰਗਦ)": a bracketed line of attribution between the source line and the verse goes with the source line
        if nxt["role"] == "text" and len(nxt.get("lines") or []) == 1 and (nxt.get("text") or "").strip().startswith("(") and i + 2 < len(regions):
            nxt = regions[i + 2]
        if r["role"] == "ref" and (r.get("parsed") or {}).get("source", "G") != "G" and nxt["role"] in ("text", "shabad") and "merged" in nxt \
                and len(nxt.get("lines") or []) >= 2:
            nxt["role"], nxt["source"] = "shabad", r["parsed"]["source"]
    if ref_before:
        # "(ਬਿਹਾਗੜਾ ਮਹਲਾ ੫)": the Granth's heading of the shabad in brackets, printed as its reference
        for r in regions:
            t = (r.get("text") or "").strip()
            if r["role"] == "heading" and t.startswith("(") and _GRANTH_HEADING.search(t) and not re.search("ਤਾਲ", t):
                r["role"], r["parsed"] = "ref", parse_ref(t)
        # under a reference lies the shabad's text, whatever the OCR made of it: a salok of two lines the
        # corpus did not match, a shabad set as prose
        for i, r in enumerate(regions[:-1]):
            nxt = regions[i + 1]
            if r["role"] == "ref" and nxt["role"] == "text" and "merged" in nxt:
                nxt["role"] = "shabad"
    # the swar-vistaar of a raag's description runs into the shabad under it: the block begins at the
    # Granth's heading of the shabad, or its first matched line; what stands before is text
    regions = _split_shabad_head(regions)
    # a grid of one line that is really a marker row or a stray
    regions = [r for r in regions if not (r["role"] == "grid" and len(r["lines"]) == 1 and r["bbox"][3] - r["bbox"][1] < body_h * 0.6)]
    # grids separated only by fragments merge; the ink decides the extent
    regions = _merge_adjacent_grids(regions, body_h)
    if ink is not None:
        regions = _ink_grids(regions, ink, page_w, page_h, body_h)
        regions = _ruled_grids(regions, ink, page_w, page_h, body_h)
    if ink is not None:
        for r in regions:
            if r["role"] == "grid":
                r["bbox"] = _ink_extent(ink, r["bbox"], page_w, page_h, body_h)
    # a page of notation has grid rows (two or more), a marker row or a section
    # label; a page of prose has long lines and none of those, and is skipped
    # by the linker; anything else (a heading and a shabad before a grid on the
    # next page) is text the next page may continue
    signals = any((r["role"] == "grid" and (len(r["lines"]) >= 2 or r.get("from_ink"))) or r["role"] in ("marker", "section") for r in regions)
    verse_lines = {n for r in regions if r["role"] == "shabad" and _verse_block(r) for n in r["lines"]}
    long_lines = sum(1 for l in lines if len((l.get("text") or "").split()) >= 6 and l.get("n") not in verse_lines)
    matched = any(r["role"] == "shabad" and any(m.get("matches") for m in (r.get("merged") or [])) for r in regions)
    if signals:
        kind = "notation"
    elif not regions:
        kind = "blank"
    elif long_lines >= 5 and not matched and not any(r["role"] == "ref" for r in regions):
        # an essay: long lines, none of them a verse the corpus knows, no reference
        kind = "prose"
    else:
        kind = "text"
    return {"page": page, "kind": kind, "regions": regions, "body_h": body_h}


_BRACKET_NUMBER = re.compile(r"^\(\s*[0-9੦-੯]{1,3}\s*\)$")
_WORD_CHAR = re.compile(r"[0-9A-Za-z\u0a05-\u0a39\u0a59-\u0a6f×°]")


def _join_number_and_taal(regions: list[dict], body_h: int) -> list[dict]:
    """
    Two heading regions on one baseline, one a bare number ("(੨)"), the other a
    taal with no raag ("ਤੀਨ ਤਾਲ"), become one numbered heading "(੨) ਤੀਨ ਤਾਲ".
    """
    out: list[dict] = []
    i = 0
    while i < len(regions):
        a = regions[i]
        b = regions[i + 1] if i + 1 < len(regions) else None
        if b and a["role"] == "heading" and b["role"] == "heading" and abs(a["bbox"][1] - b["bbox"][1]) < body_h:
            pa, pb = a.get("parsed") or {}, b.get("parsed") or {}
            num, taal = (a, b) if _BRACKET_NUMBER.match(a["text"]) else (b, a) if _BRACKET_NUMBER.match(b["text"]) else (None, None)
            pt = (taal.get("parsed") or {}) if taal else {}
            if num is not None and pt.get("taal") and not pt.get("raag") and pt.get("number") is None:
                text = "%s %s" % (num["text"].strip(), taal["text"].strip())
                out.append({"role": "heading", "bbox": _bbox_union([a["bbox"], b["bbox"]]), "lines": a["lines"] + b["lines"],
                            "text": text, "parsed": parse_heading(text)})
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def _shabad_between(regions: list[dict], body_h: int) -> list[dict]:
    out: list[dict] = []
    i = 0
    while i < len(regions):
        r = regions[i]
        out.append(r)
        if r["role"] == "heading" and not (r.get("parsed") or {}).get("section"):
            j = i + 1
            block: list[dict] = []
            while j < len(regions) and regions[j]["role"] in ("text", "shabad", "grid") and not _is_real_grid(regions[j]):
                block.append(regions[j])
                j += 1
            if block and any(b["role"] in ("shabad", "text") for b in block):
                merged = {"role": "shabad", "bbox": _bbox_union([b["bbox"] for b in block]),
                          "lines": [n for b in block for n in b["lines"]],
                          "text": "\n".join(b.get("text", "") for b in block if b.get("text")),
                          "kinds": [k for b in block for k in (b.get("kinds") or [])],
                          "merged": [m for b in block for m in (b.get("merged") or [])]}
                out.append(merged)
                i = j
                continue
        i += 1
    return out


def _fold_shreds(regions: list[dict]) -> list[dict]:
    """
    A grid of one or two OCR lines, or a text line, standing between two
    verse blocks is a line of the verse Tesseract read as something else
    (a shred of a long line, a swara-like syllable): the three become one
    shabad block. Runs until nothing folds.
    """
    changed = True
    while changed and len(regions) >= 3:
        changed = False
        for i in range(1, len(regions) - 1):
            a, b, c = regions[i - 1], regions[i], regions[i + 1]
            small = (b["role"] == "grid" and len(b["lines"]) <= 2 and not b.get("from_ink")) or \
                    (b["role"] == "text" and len(b["lines"]) <= 1)
            if a["role"] == "shabad" and c["role"] == "shabad" and small:
                merged = {"role": "shabad", "bbox": _bbox_union([a["bbox"], b["bbox"], c["bbox"]]),
                          "lines": a["lines"] + b["lines"] + c["lines"],
                          "text": "\n".join(t for t in (a.get("text"), b.get("text"), c.get("text")) if t),
                          "kinds": (a.get("kinds") or []) + (b.get("kinds") or []) + (c.get("kinds") or []),
                          "merged": (a.get("merged") or []) + (b.get("merged") or []) + (c.get("merged") or [])}
                regions = regions[:i - 1] + [merged] + regions[i + 2:]
                changed = True
                break
    return regions


# The words of a singing instruction -- a tihai or a layakari spelt out in prose
# ("ਪਹਿਲੀ ਤੋਂ ਚੌਥੀ ਮਾਤਰ ਦੇ ਟੁਕੜੇ ਨੂੰ ਦੂਜੀ ਮਾਤਰ ਤੋਂ ਲੈ ਕੇ ... ਸਮ ਤੇ ਆਉਣਾ ਹੈ।"). Its
# sentences close with a danda, so it has the shape of a verse, and in a book that
# prints the reference ABOVE the next shabad (Gurmat Sangeet Darpan) it stands right
# over that reference: the "verse over a printed reference" rule took it for the
# shabad and the notation before it lost its last lines (second cut, 2 October 2026:
# Darpan 2 p. 394, Darpan 3 p. 222). Two of these words make a block prose.
_INSTRUCTION_WORDS = re.compile(r"ਮਾਤਰ|ਟੁਕੜ|ਬਿਸਰਾਮ|ਸਮ ਤੇ|ਆਉਣਾ ਹੈ|ਮੁਕਾਅ|ਸਮਾਪਤ|ਉਚਾਰ[ਨਣ]|ਬਰਾਬਰ|ਤਿਹਾਈ|ਲੈਅਕਾਰੀ"
                                r"|ਦੁਗ[ੁ]?ਨ|ਤਿਗ[ੁ]?ਨ|ਚੌਗ[ੁ]?ਨ|ਮੁਖੜਾ|ਵੀਂ")


def _instruction(text: str) -> bool:
    """A singing instruction set as prose: two or more of its words."""
    return len({m.group(0) for m in _INSTRUCTION_WORDS.finditer(text or "")}) >= 2


def _verse_shaped(text: str) -> bool:
    """Two lines or more, most of them closed by a danda or a comma, as a Kabitt or a Savaiya is set; prose is not."""
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    if len(lines) < 2 or _instruction(text):
        return False
    closed = sum(1 for l in lines if l.rstrip("0-9੦-੯ ॥।)")[-1:] in ("।", "॥", ",") or l[-1:] in ("।", "॥", ","))
    return closed / len(lines) >= 0.5


def _granth_set(text: str, ref: dict) -> bool:
    """
    Two lines or more closed by the Granth's double danda, over a reference that names an ang: a salok
    of a vaar set in half-lines ("ਸਾਚੁ ਸੀਲ ਸਚੁ ਸੰਜਮੀ / ਸਾ ਪੂਰੀ ਪਰਵਾਰਿ॥"), which the corpus does not match
    line for line and which is not "verse-shaped" by its line ends (Guru Nanak Dev Raag Ratnaavlee, the
    Maru Vaar, pp. 106-107: three saloks and their grids read as one notation; third cut, 5 October 2026).
    """
    parsed = ref.get("parsed") or {}
    closed = sum(1 for l in (text or "").split("\n") if "॥" in l)
    return parsed.get("source", "G") == "G" and bool(parsed.get("ang_from")) and closed >= 2 and not _instruction(text)


def _verse_block(region: dict) -> bool:
    """
    A verse the corpus matched: two of its lines, or a third of them (an
    essay that quotes one line of a shabad over a page of prose is not the
    shabad), or the Granth's own heading of a shabad with a matched line.
    """
    merged = region.get("merged") or []
    matched = sum(1 for m in merged if m.get("matches"))
    if not matched:
        return False
    n = max(1, len(region.get("lines") or merged))
    return matched >= 2 or matched / n >= 0.34 or bool(_GRANTH_HEADING.search(region.get("text") or ""))


_GURMUKHI_LETTER = re.compile(r"[\u0a01-\u0a65\u0a70-\u0a75]")      # letters and their vowel signs, not the digits
_MANGAL = re.compile(r"ੴ|ਪ੍ਰਸਾਦਿ|ਸਤਿਗੁਰ|ਸਤਿ ?ਨਾਮੁ")


def _wordy(text: str) -> bool:
    """Three words of four letters or more and no held mark: a verse line, not a bol row ('ਕਬਹੂ ਖੀਰਿ ਕਾੜ ਘੀਉ ਨ ਭਾਵੈ । ਕਬਹੂ ਘਰ ਘਰ ਟੂਕ ਮਗਾਵੇ 1!')."""
    toks = [t for t in _BAR_SPLIT.split(text or "") if t]
    long_ = sum(1 for t in toks if len(_GURMUKHI_LETTER.findall(t)) >= 4)
    held = sum(1 for t in toks if token_class(t) == "held")
    return long_ >= 3 and held == 0 and not _swar_row(text)


def _fold_misread_verse(regions: list[dict], by_n: dict, body_h: int) -> list[dict]:
    """
    The last line or two of a grid region that stands right over a verse
    block, when they are lines of words or lines the corpus matched to
    the block's shabad, are the block's first lines
    (the bol-row rule of classify_line misread them). The rows of the
    notation before -- syllables, held marks, no match or the previous
    shabad's -- stay where they are.
    """
    out: list[dict] = []
    for i, r in enumerate(regions):
        nxt = regions[i + 1] if i + 1 < len(regions) else None
        if r["role"] == "grid" and nxt and nxt["role"] == "shabad" and nxt["bbox"][1] - r["bbox"][3] < body_h * 1.5:
            sid = _sid_of([nxt])
            take: list[dict] = []
            for n in reversed(r["lines"]):
                line = by_n.get(n)
                if len(take) < 2 and line and (_wordy(line.get("text") or "") or (sid is not None and _line_sid(line) == sid)):
                    take.insert(0, line)
                else:
                    break
            if take:
                rest = [n for n in r["lines"] if n not in {l.get("n") for l in take}]
                if rest:
                    out.append({**r, "lines": rest, "bbox": _bbox_union([by_n[n]["bbox"] for n in rest if by_n.get(n)]) or r["bbox"]})
                nxt.update({"lines": [l.get("n") for l in take] + nxt["lines"], "bbox": _bbox_union([l["bbox"] for l in take] + [nxt["bbox"]]),
                            "text": "\n".join([(l.get("text") or "").strip() for l in take] + [nxt["text"]]),
                            "kinds": [l.get("kind") for l in take] + nxt.get("kinds", []), "merged": take + nxt.get("merged", [])})
                continue
        out.append(r)
    return out


def _split_shabad_head(regions: list[dict]) -> list[dict]:
    """
    A shabad region whose first matched line comes after two or more
    lines the corpus does not know and that are no verse (the swar-vistaar
    of a raag's description, "ਰੇਗਮ ਪ, ਮੁ ਪਰ ਨੀਸਾ, ..."): those lines are a
    text region; the shabad keeps its Granth heading and mangal (up to
    two lines) and begins there.
    """
    out: list[dict] = []
    for r in regions:
        merged = r.get("merged") or []
        if r["role"] == "shabad" and merged:
            k = next((i for i, m in enumerate(merged) if m.get("matches")), None)
            if k is not None and k >= 2:
                keep = 0
                while keep < min(2, k):
                    t = merged[k - 1 - keep].get("text") or ""
                    if _GRANTH_HEADING.search(t) or _MANGAL.search(t):
                        keep += 1
                    else:
                        break
                cut = k - keep
                verse_like = any("॥" in (m.get("text") or "") or m.get("kind") in ("gurbani", "gurbani-unmatched") for m in merged[:cut])
                if cut >= 2 and not verse_like:
                    out.append(_region_of(merged[:cut], "text"))
                    out.append(_region_of(merged[cut:], "shabad"))
                    continue
        out.append(r)
    return out


def _is_real_grid(region: dict) -> bool:
    """A grid region of two OCR lines or more (a swar row over a bol row) is a grid; one line under a heading is a shred."""
    return region["role"] == "grid" and (len(region["lines"]) >= 2 or region.get("from_ink"))


def _looks_like_verse(text: str) -> bool:
    return bool(re.search("[।॥]", text)) and len(text.split()) >= 3


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def _marker_row(text: str) -> list[str | None]:
    """'° x ° x' -> ['0', '×', '0', '×']; a token that is no marker is None."""
    out = []
    for t in text.split():
        k = marker_kind(t)
        if k == "sam" or t in ("x", "X", "ਨ", "+"):
            out.append("×")
        elif k == "khali" or t in ("°", "o", "O", "[o)", "০", "(o)"):
            out.append("0")
        elif k == "tali":
            out.append(gurmukhi_digits(t))
        elif t in ("ਤੇ", "ੜੇ", "੩"):
            out.append("3")
        else:
            out.append(None)
    return out


def _ink_grids(regions: list[dict], ink, page_w: int, page_h: int, body_h: int) -> list[dict]:
    """
    A ruled table Tesseract read nothing of leaves a hole in the regions: a
    section label or a heading, then a tall gap full of ink, then the next
    line. The ink in such a gap is a grid region with no OCR lines; the
    reader works from the ink anyway.
    """
    import numpy as np
    edge = int(page_w * 0.04)
    out: list[dict] = []
    ordered = sorted(regions, key=lambda r: r["bbox"][1])
    for i, r in enumerate(ordered):
        out.append(r)
        # only under a section label, or a heading that names a taal: a paragraph of prose has gaps too
        if not (r["role"] == "section" or (r["role"] == "heading" and (r.get("parsed") or {}).get("taal"))):
            continue
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        ya = r["bbox"][3] + 4
        yb = (nxt["bbox"][1] - 4) if nxt else int(page_h * 0.93)
        if yb - ya < 2.5 * body_h:
            continue
        band = ink[ya:yb, edge:page_w - edge]
        if band.size == 0:
            continue
        rows_ink = np.where((band > 0).sum(axis=1) > 2)[0]
        if len(rows_ink) < 1.5 * body_h:
            continue                                  # a real gap, not a table
        cols_ink = np.where((band > 0).sum(axis=0) > 2)[0]
        bbox = [int(edge + cols_ink.min()), int(ya + rows_ink.min()), int(edge + cols_ink.max()) + 1, int(ya + rows_ink.max()) + 1]
        if (bbox[2] - bbox[0]) < 0.3 * page_w:
            continue
        out.append({"role": "grid", "bbox": bbox, "lines": [], "text": "", "kinds": [], "merged": [], "from_ink": True})
    return sorted(out, key=lambda r: r["bbox"][1])


def _ruled_tables(ink, page_w: int, page_h: int, body_h: int) -> list[tuple[int, int]]:
    """
    The row bands [(y0, y1)] of the page's ruled tables: two or more vertical rules, each ten lines
    tall or more, standing over the same rows -- the bars that divide a notation's vibhags from the
    table's top rule to its bottom one. A rule at the page's edge (a border, a scanner's shadow) and
    a band as tall as the page (a frame round the whole page) are not a table.
    """
    from lib.ocr_grid import vertical_rules
    edge = int(page_w * 0.06)
    rules = [r for r in vertical_rules(ink, int(10 * body_h)) if edge < r["x"] < page_w - edge]
    rules.sort(key=lambda r: r["y0"])
    bands: list[list] = []                                      # [y0, y1, how many rules]
    for r in rules:
        for b in bands:
            shared = min(b[1], r["y1"]) - max(b[0], r["y0"])
            if shared >= 0.6 * min(b[1] - b[0], r["y1"] - r["y0"]):
                b[0], b[1], b[2] = min(b[0], r["y0"]), max(b[1], r["y1"]), b[2] + 1
                break
        else:
            bands.append([r["y0"], r["y1"], 1])
    return [(b[0], b[1]) for b in bands if b[2] >= 2 and b[1] - b[0] < 0.9 * page_h]


def _ruled_grids(regions: list[dict], ink, page_w: int, page_h: int, body_h: int) -> list[dict]:
    """
    A notation set in a ruled table is as tall as its bars, whatever the OCR read of it. Tesseract
    returns a third of such a table, or nothing (Prof Tara Singh's Bhagat Hayt Gavai Ravidasa, p. 96:
    the sthai read, the antra not; p. 51: a blank page; third cut, 5 October 2026). Where the bars
    stand the grid stands: the grid regions among them are grown to the bars' top and bottom, and bars
    with no grid region among them make one, read from the ink. A table that holds a verse or a
    reference is no notation's and is left alone.
    """
    import numpy as np
    for ya, yb in _ruled_tables(ink, page_w, page_h, body_h):
        def inside(r):
            return ya - body_h < (r["bbox"][1] + r["bbox"][3]) / 2 < yb + body_h
        held = [r for r in regions if inside(r)]
        if any(r["role"] in ("shabad", "ref") for r in held):
            continue
        # the shreds Tesseract read inside the table (a column of specks down a bar) are the grid's own
        shreds = [r for r in held if r["role"] == "text" and len(r.get("lines") or []) <= 2]
        regions = [r for r in regions if not any(r is s for s in shreds)]
        held = [r for r in held if not any(r is s for s in shreds)]
        grids = [r for r in held if r["role"] == "grid"]
        if grids:
            top, low = min(grids, key=lambda r: r["bbox"][1]), max(grids, key=lambda r: r["bbox"][3])
            if low["bbox"][3] < yb and not any(r is not low and r["bbox"][1] > low["bbox"][3] for r in held):
                low["bbox"][3] = int(yb)
                low["from_ink"] = True
            if top["bbox"][1] > ya and not any(r is not top and r["bbox"][3] < top["bbox"][1] for r in held):
                top["bbox"][1] = int(ya)
                top["from_ink"] = True
            continue
        edge = int(page_w * 0.04)
        cols = np.where((ink[ya:yb, edge:page_w - edge] > 0).sum(axis=0) > 2)[0]
        if len(cols) == 0 or cols.max() - cols.min() < 0.3 * page_w:
            continue
        regions.append({"role": "grid", "bbox": [int(edge + cols.min()), int(ya), int(edge + cols.max()) + 1, int(yb)],
                        "lines": [], "text": "", "kinds": [], "merged": [], "from_ink": True})
    return sorted(regions, key=lambda r: r["bbox"][1])


def _merge_adjacent_grids(regions: list[dict], body_h: int) -> list[dict]:
    out: list[dict] = []
    for r in regions:
        if r["role"] == "grid" and out and out[-1]["role"] == "grid" and r["bbox"][1] - out[-1]["bbox"][3] < body_h * 1.5:
            out[-1]["lines"] += r["lines"]
            out[-1]["bbox"] = _bbox_union([out[-1]["bbox"], r["bbox"]])
        elif r["role"] == "text" and out and out[-1]["role"] == "grid" and len(r["lines"]) <= 2 and len(r["text"].split()) <= 4:
            out[-1]["lines"] += r["lines"]
            out[-1]["bbox"] = _bbox_union([out[-1]["bbox"], r["bbox"]])
        else:
            out.append(r)
    # a grid followed by text fragments then grid again: merge across
    merged: list[dict] = []
    for r in out:
        if r["role"] == "grid" and len(merged) >= 2 and merged[-1]["role"] == "text" and merged[-2]["role"] == "grid" \
                and len(merged[-1]["lines"]) <= 2 and r["bbox"][1] - merged[-2]["bbox"][3] < body_h * 4:
            frag = merged.pop()
            merged[-1]["lines"] += frag["lines"] + r["lines"]
            merged[-1]["bbox"] = _bbox_union([merged[-1]["bbox"], frag["bbox"], r["bbox"]])
        else:
            merged.append(r)
    return merged


def _ink_extent(ink, bbox: list[int], page_w: int, page_h: int, body_h: int) -> list[int]:
    """The grid's real box: the OCR lines' box grown to the ink rows and columns it touches."""
    import numpy as np
    x0, y0, x1, y1 = bbox
    pad_y = int(body_h * 0.6)
    ya, yb = max(0, y0 - pad_y), min(page_h, y1 + pad_y)
    band = ink[ya:yb, :]
    if band.size == 0:
        return bbox
    cols = np.where((band > 0).sum(axis=0) > 0)[0]
    rows = np.where((band > 0).sum(axis=1) > 0)[0]
    if len(cols) == 0 or len(rows) == 0:
        return bbox
    # ignore ink within 3% of the page edges (scanner shadow)
    edge = int(page_w * 0.03)
    cols = cols[(cols > edge) & (cols < page_w - edge)]
    if len(cols) == 0:
        return bbox
    return [int(cols.min()), int(ya + rows.min()), int(cols.max()) + 1, int(ya + rows.max()) + 1]


# ---- the book ---------------------------------------------------------------

def _running_header(region: dict, page_h: int) -> bool:
    """
    The book's running header, not a notation's heading: 'ਰਾਗ ਜੈਤਸਰੀ' alone
    at the top of the page, or 'ਰਾਗ ਰਾਮਕਲੀ ਰ 299' / '70 ਸ੍ਰੀ ਗੁਰੂ ਤੇਗ ਬਹਾਦਰ ਰਾਗ
    ਰਤਨਾਵਲੀ' -- a raag with no taal, a page number at either end.
    """
    p = region.get("parsed") or {}
    if p.get("taal") or p.get("section"):
        return False
    top = page_h and region["bbox"][1] < 0.1 * page_h
    if not top or not p.get("raag"):
        return False
    text = (region.get("text") or "").strip()
    if p.get("number") is None and len(text.split()) <= 3:
        return True
    return bool(re.match(r"^[0-9੦-੯]{1,4}\b", text) or re.search(r"\b[0-9੦-੯]{1,4}\s*$", text))


def _shabad_heading(region: dict) -> bool:
    """'ਜੈਤਸਰੀ ਮਹਲਾ ੫ ॥': the Granth's own heading of the shabad, part of its text."""
    t = (region.get("text") or "")
    return bool(_GRANTH_HEADING.search(t)) and not re.search("ਤਾਲ", t)


def raag_descriptions(layouts: list[dict]) -> list[dict]:
    """
    The prose a book prints about a raag -- its heading names the raag and
    no taal, and what follows is text until the next heading, section label
    or grid. [{"raag": parsed raag, "heading": region, "page", "pages",
    "regions": [(page, region)]}]. A description may run over a page turn;
    it ends at the first notation signal.
    """
    out: list[dict] = []
    cur: dict | None = None
    prev_page = None
    for lay in layouts:
        page = lay["page"]
        if prev_page is not None and page != prev_page + 1 and cur:
            out.append(cur); cur = None
        prev_page = page
        regions = lay["regions"]
        for k, r in enumerate(regions):
            role = r["role"]
            p = r.get("parsed") or {}
            # a running-header-shaped raag heading at the top of the page is the description's when the
            # description's own heading ("ਰਾਗੁ ਪਰੀਚੈ:-") or its words stand right under it
            opens = len(_description_words(regions[k + 1:k + 8])) >= 2
            if role == "heading" and p.get("raag") and not p.get("taal") and not p.get("section") and not _shabad_heading(r) \
                    and (opens or not _running_header(r, lay.get("page_h") or 0)):
                if cur and (cur["raag"] or {}).get("key") and (cur["raag"] or {}).get("key") == p["raag"].get("key"):
                    # a sentence of the same description that names the raag again: not a new one
                    if page not in cur["pages"]:
                        cur["pages"].append(page)
                    cur["regions"].append((page, r))
                    continue
                if cur:
                    out.append(cur)
                cur = {"raag": p["raag"], "heading": r, "page": page, "pages": [page], "regions": []}
                continue
            if cur is None:
                continue
            if role in ("grid", "marker") and not cur["regions"]:
                continue                                   # swar lines under the heading read as a grid: the description has not begun
            if role in ("text", "shabad", "note", "ref") or (role == "heading" and _describes_raag(r)):
                if page not in cur["pages"]:
                    cur["pages"].append(page)
                cur["regions"].append((page, r))         # "ਰਾਗੁ ਪਰੀਚੈ:-", "ਥਾਟ-ਬਿਲਾਵਲ": the description's own headings
            else:
                out.append(cur); cur = None
    if cur:
        out.append(cur)
    # a heading with nothing under it is a running header of another kind, not a description; a heading
    # whose taal the OCR lost, over a shabad and its reference, is a notation's (its prose is nowhere)
    def prose(d):
        return sum(len(r.get("lines") or []) for _, r in d["regions"] if r["role"] in ("text", "note") or _describes_raag(r))
    return [d for d in out if prose(d) >= 2]


def _label_sections(span: dict) -> None:
    """
    A section with no printed label: the first of a span is the sthai (a
    marker row or a grid opens the notation straight after the shabad, or a
    page turn separated the label from its grid); a later one stays unknown.
    """
    for k, sec in enumerate(span["sections"]):
        if sec.get("kind"):
            continue
        sec["assumed"] = True
        if k == 0:
            sec["kind"], sec["n"] = "sthai", 1


LONG_SPAN_PAGES = 5         # a shabad's notation runs over two or three pages; past five it is flagged for the reviewer's eye, never cut
CONTENT_ROLES = ("grid", "section", "marker", "note", "text")     # what makes a shabad's span a notation
NOTATION_ROLES = ("grid", "section", "marker")                   # what says the grids have begun


def _items(layouts: list[dict], style: dict | None = None) -> list[dict]:
    """
    Every region of the pages read, in reading order, as {"page", "role",
    "region", "first"} -- `first` marks the first region of a page. A gap in
    the pages read and a page of prose (an essay on a raag, a preface) are
    breaks: {"role": "break"}; nothing links across them. The running
    header is dropped and the Granth's own heading of a shabad (ਗਉੜੀ ਮਹਲਾ ੫)
    is part of the shabad.
    """
    items: list[dict] = []
    repeated = _repeated_tops(layouts, style)
    prev = None
    for lay in layouts:
        page = lay["page"]
        page_h = lay.get("page_h") or 0
        if (prev is not None and page != prev + 1) or lay.get("kind") == "prose":
            items.append({"page": page, "role": "break", "region": None, "first": True})
        prev = page
        if lay.get("kind") == "prose":
            continue
        first = True
        for r in lay["regions"]:
            role = r["role"]
            if role == "heading" and _running_header(r, page_h):
                continue
            if first and role in ("heading", "text") and _is_running(r, page_h, repeated):
                continue                                   # "424. ਗੁਰੂ ਅਰਜਨ ਦੇਵ ਰਾਗ ਰਤਨਾਵਲੀ": the book's running header
            if role == "heading" and _shabad_heading(r):
                role = "shabad"
                r = {**r, "role": "shabad", "kinds": ["heading"], "merged": r.get("merged") or []}
            items.append({"page": page, "role": role, "region": r, "first": first,
                          "top": bool(page_h and r["bbox"][1] < 0.1 * page_h)})
            first = False
    return items


_AT_NUMBER = re.compile("(?:ਸ਼ਬਦ|ਸਬਦ)\\s*ਨੰ")
_DIGITS_RE = re.compile(r"[0-9੦-੯.:।()\-]+")


def _top_key(region: dict, page_h: int) -> str | None:
    """The words of a one-line region in the top tenth of the page, digits dropped; None lower down."""
    if not page_h or region["bbox"][1] > 0.1 * page_h or len(region.get("lines") or []) > 1:
        return None
    words = _DIGITS_RE.sub(" ", region.get("text") or "").split()
    return " ".join(words) if len(words) >= 2 else None


def _similar(a: str, b: str) -> bool:
    from difflib import SequenceMatcher
    return bool(a and b) and SequenceMatcher(None, a, b).ratio() >= 0.6


def _repeated_tops(layouts: list[dict], style: dict | None = None) -> list[str]:
    """
    The top-of-page texts that recur -- on two pages of a short read (five or
    fewer), on three of a longer one -- clustered by similarity, since OCR
    spells the same header differently from page to page; plus whatever the
    manifest names as the book's running header (style.running_header).
    """
    clusters: list[list] = []            # [text, count]
    n = 0
    for lay in layouts:
        n += 1
        regs = lay.get("regions") or []
        if not regs:
            continue
        key = _top_key(regs[0], lay.get("page_h") or 0)
        if not key:
            continue
        for c in clusters:
            if _similar(c[0], key):
                c[1] += 1
                break
        else:
            clusters.append([key, 1])
    need = 3 if n > 5 else 2
    out = [c[0] for c in clusters if c[1] >= need]
    named = (style or {}).get("running_header")
    if named:
        out += [named] if isinstance(named, str) else list(named)
    return out


def _header_line(text: str, named: list[str]) -> bool:
    """A line that is one of the manifest's running headers, its page number and dandas aside: nearly letter for letter, since a verse may resemble a header."""
    from difflib import SequenceMatcher
    key = " ".join(_DIGITS_RE.sub(" ", re.sub("[॥।]", " ", text)).split())
    return bool(key) and any(SequenceMatcher(None, key, h).ratio() >= 0.85 for h in named)


def _is_running(region: dict, page_h: int, repeated: list[str]) -> bool:
    """A top-of-page region that reads like a repeated header -- never one that names a taal: Dyal Singh's consecutive numbered headings resemble each other too."""
    if (region.get("parsed") or {}).get("taal"):
        return False
    key = _top_key(region, page_h)
    return bool(key) and any(_similar(key, r) for r in repeated)


def _sid_of(regions: list[dict]) -> int | None:
    """The shabad the corpus matches on these regions' lines vote for, or None."""
    from lib.notation_resolve import votes_from_lines
    weight, _, _ = votes_from_lines([m for r in regions for m in (r.get("merged") or [])])
    return max(weight, key=weight.get) if weight else None


def _line_sid(line: dict) -> int | None:
    """The shabad one merged line's best corpus match names, or None."""
    best = None
    for m in line.get("matches") or []:
        if m.get("shabad_id") is not None and (best is None or float(m.get("score") or 0) > float(best.get("score") or 0)):
            best = m
    return best.get("shabad_id") if best else None


_SWAR_TOKEN = re.compile(r"^(?:[ਸਰਗਮਪਧਨ][ਾਿੀੁੂੇੈੋੌੰਂ\u0a3c]*){1,4}$|^[-—–ऽsS5]+$")


def _swar_row(text: str) -> bool:
    """'ਪ ਸੀ ਰੋ ਸਾ ਧਧ ਪ ਮਪ ਧਨੀ': three tokens or more, most of them swara letters (or held marks) and nothing else."""
    toks = [t for t in _BAR_SPLIT.split(text or "") if t]
    if len(toks) < 3:
        return False
    return sum(1 for t in toks if _SWAR_TOKEN.match(t)) >= 0.6 * len(toks)


def _region_of(lines: list[dict], role: str) -> dict:
    boxes = [m["bbox"] for m in lines if m.get("bbox")]
    return {"role": role, "merged": lines, "lines": [m.get("n") for m in lines],
            "text": "\n".join((m.get("text") or "").strip() for m in lines),
            "kinds": [m.get("kind") for m in lines], "bbox": _bbox_union(boxes) if boxes else [0, 0, 0, 0]}


def _trim_prev_bol(region: dict, prev_sid: int | None) -> tuple[dict, dict | None]:
    """
    A verse block headed by the previous notation's last rows -- its bol row
    read as a verse (the syllables spell the words the corpus knows), its
    swar rows read as nothing the corpus knows -- begins at its first line
    of its own. Returns (the block from there, the rows cut off as a grid
    region for the notation before, or None).
    """
    merged = region.get("merged") or []
    if prev_sid is None or len(merged) < 2:
        return region, None
    sids = {_line_sid(m) for m in merged} - {None}
    if sids and sids <= {prev_sid}:
        return region, None             # the same shabad throughout: set again, or its next pada -- the segmenter decides
    k = 0
    while k < len(merged) - 1:
        line = merged[k]
        sid = _line_sid(line)
        if sid == prev_sid:
            k += 1
        elif sid is None and not _GRANTH_HEADING.search(line.get("text") or "") and not _wordy(line.get("text") or "") \
                and (classify_line(line) in ("grid", "marker") or _swar_row(line.get("text") or "")
                     or not re.search(r"[\u0a05-\u0a39\u0a59-\u0a5e]", line.get("text") or "")):     # '[ |': a bar read alone
            k += 1
        else:
            break
    if k == 0:
        return region, None
    return _region_of(merged[k:], "shabad") | {"role": "shabad"}, _region_of(merged[:k], "grid")


def _named_heading(region: dict) -> dict | None:
    """The parse of a heading that names a raag or a taal (a notation's heading, not a section label)."""
    p = region.get("parsed") or {}
    return p if (p.get("raag") or p.get("taal")) and not p.get("section") else None


_ORDINALS = {"ਪਹਿਲਾ": 1, "ਪਹਿਲੀ": 1, "ਦੂਜਾ": 2, "ਦੂਜੀ": 2, "ਤੀਜਾ": 3, "ਤੀਜੀ": 3, "ਚੌਥਾ": 4, "ਚੌਥੀ": 4, "ਪੰਜਵਾਂ": 5, "ਪੰਜਵੀਂ": 5}


def _section_heading(region: dict, grids_seen: bool) -> dict | None:
    """
    A heading that is a section of the notation rather than a notation's own:
    'ਅੰਤਰਾ-ਸੂਲਫਾਕ (ਮਣੀ ਤਾਲ ਦੀ ਚੌਗੁਨ ਲਯ ਵਿਚ)', 'ਦੂਜਾ ਅੰਤਰਾ-ਤਾਲ ਰੂਪਕ', or -- once
    the grids have begun -- 'ਤਾਲ ਚੰਚਲ (...)' alone (a partaal changes taal
    from section to section). {"kind", "n", "taal"} or None.
    """
    p = region.get("parsed") or {}
    if p.get("section"):
        return {"kind": p["section"].get("kind"), "n": p["section"].get("n"), "taal": p.get("taal")}
    words = [w for w in re.split(r"[\s\-:–—(),]+", (region.get("text") or "").strip()) if w]
    if words:
        n = None
        k = 0
        if words[0] in _ORDINALS and len(words) > 1:
            n, k = _ORDINALS[words[0]], 1
        lab = section_label(words[k]) if k < len(words) else None
        if lab:
            return {"kind": lab[0], "n": lab[1] if lab[1] is not None else n, "taal": p.get("taal")}
    if grids_seen and p.get("taal") and not p.get("raag") and p.get("number") is None:
        return {"kind": None, "n": None, "taal": p["taal"]}
    return None


def _opens_notation(region: dict) -> bool:
    """A heading that names a taal or carries a notation number: it opens a notation."""
    p = _named_heading(region)
    return bool(p and (p.get("taal") or p.get("number") is not None))


def _is_lead(item: dict) -> bool:
    """A heading, or a bare number on a line of its own ("੧੮."), standing between one notation and the next shabad leads the next in; a loose line of text is the tail of the one before."""
    if item["role"] == "heading":
        return True
    if item["role"] != "text" or len(item["region"].get("lines") or []) > 1:
        return False
    text = (item["region"].get("text") or "").strip()
    if _NUMBER_LINE.match(text):
        return True
    # "ਗੋਂਡ" alone over the verse: the shabad's raag as the Granth titles it, set as a line of its own
    raag = parse_heading(text).get("raag") if len(text.split()) <= 2 else None
    return bool(raag and raag.get("key") and (raag.get("confidence") or 0) >= 0.9)


def _grids_after_shabad(seg: dict, real: bool = False) -> bool:
    """A grid stands after the span's last shabad or reference line: the notation proper has begun (`real`: a grid of two rows or more, not a one-line shred under the verse)."""
    last = -1
    for k, it in enumerate(seg["items"]):
        if it["role"] in ("shabad", "ref"):
            last = k
    return any(it["role"] == "grid" and (not real or _is_real_grid(it["region"])) for it in seg["items"][last + 1:])


def _segments(items: list[dict], after: bool, pairing: bool = False) -> list[dict]:
    """
    The items cut into notations. A notation is a shabad and what the book
    prints after it (before it, in a book that sets the shabad after the
    grids) up to the next shabad: the next shabad's lead-in -- its heading,
    its number -- goes with the next; a shabad the corpus links to the same
    shabad as the open notation (a verse printed before each of its grids)
    continues it. A heading that names a raag or a taal after a finished
    notation opens the next; a raag heading followed by prose (a
    description) leaves the description to be dropped later, shabadless.
    """
    segs: list[dict] = []
    cur: dict | None = None

    def close():
        nonlocal cur
        if cur and cur["items"]:
            segs.append(cur)
        cur = None

    def open_(**kw):
        return {"items": [], "sid": None, "after_break": False, "opened_by": None, **kw}

    def has(seg, roles):
        return any(it["role"] in roles for it in seg["items"])

    i = 0
    broke = True
    while i < len(items):
        it = items[i]
        role = it["role"]
        if role == "break":
            close()
            broke = True
            i += 1
            continue
        if cur is None:
            cur = open_(after_break=broke)
            broke = False
        if role == "shabad":
            group = []
            j = i
            while j < len(items) and items[j]["role"] in ("shabad", "ref"):
                group.append(items[j])
                j += 1
            has_ref = any(g["role"] == "ref" for g in group)
            nxt_role = items[j]["role"] if j < len(items) else "break"
            if cur.get("sid") is not None and has(cur, ("grid",)):
                first = group[0]
                trimmed, tail = _trim_prev_bol(first["region"], cur["sid"])
                if tail is not None:
                    cur["items"].append({**first, "role": "grid", "region": tail})     # the rows are the notation before's
                    if _sid_of([trimmed]) is None and not has_ref and not _GRANTH_HEADING.search(trimmed.get("text") or ""):
                        # what is left is no verse ("ao", "ਅਤਰਾ"): the rows and the label are the open notation's
                        cur["items"].append({**first, "role": "text", "region": trimmed, "first": False})
                        i = j
                        continue
                    group[0] = {**first, "region": trimmed, "first": False}
            sid = _sid_of([g["region"] for g in group if g["role"] == "shabad"])
            if after:
                cur["items"].extend(group)
                cur["sid"] = cur["sid"] or sid
                close()
                i = j
                continue
            same = sid is not None and sid == cur["sid"]
            begun = has(cur, ("shabad",)) and _grids_after_shabad(cur, real=True)
            if begun and not has_ref and len(group) == 1 and len(group[0]["region"].get("lines") or []) == 1 \
                    and nxt_role in ("grid", "marker", "section", "text"):
                # one line of verse between the grids, no reference, grids again under it: the antra's
                # first words set as its label ("ਹਰਿ ਹਰਿ ਅਗਮ ਅਗਾਧੋ ॥"), not the next shabad
                cur["items"].append({**group[0], "role": "text"})
                i = j
                continue
            # the same shabad set again after its reference under a heading with a taal or a number: a
            # chhant's next pada, a notation of its own (a verse printed before each grid has no heading)
            again = same and begun and has_ref and nxt_role == "heading" and _opens_notation(items[j]["region"])
            # prose before the first shabad (a raag's description, its swar-vistaar) is not the notation's
            prose_before = not has(cur, ("shabad",)) and not has(cur, NOTATION_ROLES) \
                and any(it["role"] == "text" and len(it["region"].get("lines") or []) >= 3 for it in cur["items"])
            if (has(cur, ("shabad",)) and not same) or again or (not has(cur, ("shabad",)) and has(cur, NOTATION_ROLES)) or prose_before:
                # the next shabad -- or the first, after grids whose shabad was not read (the previous page, a failed read)
                lead: list[dict] = []
                # a reference standing after the grids is the next shabad's, printed above it (Gurmat
                # Sangeet Darpan sets "(ਕਾਨੜੇ ਕੀ ਵਾਰ ਮਹਲਾ ੪) (੧੩੧੮)" over the salok)
                while cur["items"] and (_is_lead(cur["items"][-1])
                                        or (cur["items"][-1]["role"] == "ref" and has(cur, ("grid",)))):
                    lead.insert(0, cur["items"].pop())
                close()
                cur = open_(items=lead, opened_by="shabad")
            cur["items"].extend(group)
            if cur["sid"] is None:
                cur["sid"] = sid
            i = j
            continue
        if role == "heading":
            intro = _describes_raag(it["region"])                     # "ਰਾਗੁ ਪਰੀਚੈ:-", "ਥਾਟ - ਭੈਰਵ।"
            if not intro and _section_heading(it["region"], has(cur, ("grid",))):
                cur["items"].append(it)                 # a section's own heading (ਅੰਤਰਾ, a partaal's next taal): inside the notation
                i += 1
                continue
            # a description's own line names no taal, whatever the fuzzy match made of its words ("ਰਾਗੁ-ਤੁਖਾਰੀ
            # ਥਾਟ-ਤੋੜੀ ਸਵਰ-ਦੋਨੋਂ ਨਿਸ਼ਾਦ" read as a sawari heading rode along with the notation before it)
            p = {} if intro else (_named_heading(it["region"]) or {})
            if not p and not intro and (it["region"].get("parsed") or {}).get("number") is not None:
                p = {"number": it["region"]["parsed"]["number"]}        # "(੨)" alone: the taal beside it lost by the OCR
            # "ਰਾਗ ਨਦ ਕੱਸ": a heading that names a raag the vocabulary cannot read (OCR) still names one
            names_raag = bool(p.get("raag")) or bool(_RAAG_WORD.match(it["region"].get("text") or ""))
            if p or intro or names_raag:
                # the notation before it is done: grids after its shabad, or (a body whose shabad the book
                # printed elsewhere: Padhti Granth's second kirtankar's "(੨) ਤੀਨਤਾਲ") grids with no shabad at all
                done = (has(cur, ("shabad",)) and _grids_after_shabad(cur)) \
                    or (not has(cur, ("shabad",)) and any(it["role"] == "grid" and _is_real_grid(it["region"]) for it in cur["items"]))
                # "13. ਰਾਗ ਰਾਮਕਲੀ" over the raag's vaadi, aaroh and swar-vistaar is a chapter of the book, not a
                # notation's number (Bhagat Hayt Gavai Ravidasa: the description took the shabad before it)
                chapter = names_raag and not p.get("taal") and p.get("number") is not None and _description_follows(items, i)
                raag_only = names_raag and not p.get("taal") and (p.get("number") is None or chapter)
                nxt = items[i + 1]["role"] if i + 1 < len(items) else "break"
                # a raag's heading followed by prose, below the top of the page, opens a description;
                # at the top of a page it is a running header the repetition test missed -- unless what
                # follows says the words of a description (vaadi, thaat, pakad, aaroh), or the heading
                # itself is the description's ("ਰਾਗੁ ਪਰੀਚੈ")
                described = intro or ((raag_only or not p) and _description_follows(items, i))
                description = (raag_only and (p.get("raag") or {}).get("key") and not it.get("top") and nxt in ("text", "break")) \
                    or ((raag_only or intro) and described)
                # a heading with a taal or a number after the grids is the next notation's; a raag alone is
                # not a boundary (Dyal Singh's Hindustani raag headings hold shabads of several Granth raags)
                if after or (done and p and not raag_only) or (description and (has(cur, NOTATION_ROLES) or pairing)):
                    # the raag's heading at the top of the page above its "ਰਾਗੁ ਪਰੀਚੈ" leads the description, not the notation
                    lead = []
                    while cur["items"] and cur["items"][-1]["role"] == "heading" and _is_lead(cur["items"][-1]):
                        lead.insert(0, cur["items"].pop())
                    close()
                    cur = open_(items=lead, opened_by="raag" if (raag_only or intro) else "heading")
            cur["items"].append(it)
            i += 1
            continue
        cur["items"].append(it)
        i += 1
    close()
    return segs


def _pair_by_section(segs: list[dict]) -> list[dict]:
    """
    style.pairing == "section" (Guru Nanak Sangeet Padhti Granth): the book is
    organised by raag, a section opening at the raag's heading and description,
    and within a section it sets its shabads and their notations in the same
    order but not together -- one shabad then two numbered notations of it by
    two kirtankars, two shabads then two notations, a notation then its shabad
    under it. So within a section the k-th body (a segment with grids) takes
    the k-th shabad block (the last one when the shabads run out), whatever
    verse happens to stand inside the body; the shabad blocks themselves are
    consumed, and a raag's description at the head of a body is left out.
    """
    out: list[dict] = []
    section: list[dict] = []

    def blocks(sg):
        # the shabad blocks of a segment, one per shabad: two shabads listed one under the other are
        # consecutive shabad items the linker grouped as one, told apart by the corpus votes on each
        out_: list[tuple] = []
        block: list[dict] = []
        bsid = None
        for it in sg["items"]:
            if it["role"] == "shabad":
                sid = _sid_of([it["region"]])
                if block and sid is not None and bsid is not None and sid != bsid:
                    out_.append((bsid, block))
                    block = []
                block.append(it)
                bsid = sid if sid is not None else bsid
            elif it["role"] == "ref" and block:
                block.append(it)
            elif block:
                out_.append((bsid, block))
                block, bsid = [], None
        if block:
            out_.append((bsid, block))
        return out_

    def flush():
        if not section:
            return
        shabads = [b for sg in section for b in blocks(sg)]
        # a body holds a notation proper: a section label (ਸਥਾਈ) or a marker row; a description's aaroh
        # and avroh read as a grid of two lines and are neither
        bodies = [sg for sg in section if any(it["role"] in ("section", "marker") for it in sg["items"])]
        if not shabads or not bodies:
            out.extend(section)
            section.clear()
            return
        for k, sg in enumerate(bodies):
            sid, sh = shabads[min(k, len(shabads) - 1)]
            rest = [it for it in sg["items"] if it["role"] not in ("shabad", "ref")]
            if sg.get("opened_by") == "raag":
                # the raag's heading and description lead the body: not the notation's
                first = next((n for n, it in enumerate(rest) if it["role"] in NOTATION_ROLES or (it["role"] == "heading" and _opens_notation(it["region"]))), 0)
                rest = rest[first:]
            own = any(it["role"] == "shabad" for it in sg["items"])
            paired_own = own and sh and any(it is sh[0] for it in sg["items"])
            sg["items"] = list(sh) + rest
            sg["sid"] = sid
            sg["paired"] = True
            if not paired_own:
                sg["inherited"] = True
            out.append(sg)
        section.clear()

    for sg in segs:
        # a raag's heading opens a section once the one before holds a notation (a description may run
        # over several headings: ਰਾਗੁ ਭੈਰਉ, then ਥਾਟ - ਭੈਰਵ)
        if sg.get("after_break") or (sg.get("opened_by") == "raag"
                                     and any(it["role"] in NOTATION_ROLES for s2 in section for it in s2["items"])):
            flush()
        section.append(sg)
    flush()
    return out


def _span_from_items(seg: dict) -> dict:
    """The span dict the parser reads, from a segment's items."""
    span = {"heading": None, "heading_page": None, "headings": [], "shabad": [], "ref": None, "sections": [],
            "notes": [], "text": [], "pages": [], "continued": False, "continues": False, "extent": {},
            "sid": seg.get("sid"), "inherited": bool(seg.get("inherited")),
            "long": len({it["page"] for it in seg["items"]}) > LONG_SPAN_PAGES}

    def touch(page):
        if page not in span["pages"]:
            span["pages"].append(page)

    def section(page):
        if not span["sections"]:
            span["sections"].append({"label": None, "kind": None, "n": None, "page": page, "grids": [], "markers": [], "taal": None})
        return span["sections"][-1]

    for it in seg["items"]:
        page, r, role = it["page"], it["region"], it["role"]
        touch(page)
        ext = span["extent"].get(page)
        span["extent"][page] = _bbox_union([ext, r["bbox"]]) if ext else list(r["bbox"])
        if role == "heading":
            p = r.get("parsed") or {}
            sec = _section_heading(r, bool(span["sections"]) and any(s["grids"] for s in span["sections"]))
            if sec:
                span["sections"].append({"label": r, "kind": sec.get("kind"), "n": sec.get("n"), "page": page,
                                         "grids": [], "markers": [], "taal": sec.get("taal")})
            else:
                span["headings"].append((page, r))
                if span["heading"] is None:
                    span["heading"], span["heading_page"] = r, page
        elif role == "section":
            sec = r.get("parsed") or {}
            span["sections"].append({"label": r, "kind": sec.get("kind"), "n": sec.get("n"), "page": page,
                                     "grids": [], "markers": [], "taal": None})
        elif role == "marker":
            section(page)["markers"].append((page, r))
        elif role == "grid":
            section(page)["grids"].append((page, r))
        elif role == "shabad":
            span["shabad"].append((page, r))
        elif role == "ref":
            if span["ref"] is None:
                span["ref"] = r
        elif role == "note":
            span["notes"].append((page, r))
        elif role == "text":
            span["text"].append((page, r))
    # the heading's facts may stand on two lines (the raag, then "ਤਾਲ ਤੀਨਤਾਲ"): one heading, filled from all
    if len(span["headings"]) > 1:
        first = span["heading"]
        parsed = dict(first.get("parsed") or {})
        for _, r in span["headings"][1:]:
            for k, v in (r.get("parsed") or {}).items():
                if v and not parsed.get(k):
                    parsed[k] = v
        span["heading"] = {**first, "parsed": parsed, "text": " / ".join(r["text"] for _, r in span["headings"])}
    span["pages"].sort()                   # a paired shabad's page may come after its notation's
    first_item = seg["items"][0]
    span["continued"] = bool(seg.get("after_break") is False and first_item["first"]
                             and first_item["role"] in ("grid", "marker", "section")) or bool(seg.get("continued"))
    _label_sections(span)
    return span


def header_page_numbers(layouts: list[dict], style: dict | None = None) -> dict[int, int]:
    """
    {scan page: printed page number} from the running headers ('70 ਸ੍ਰੀ ਗੁਰੂ
    ਤੇਗ ਬਹਾਦਰ ਰਾਗ ਰਤਨਾਵਲੀ', 'ਰਾਗ ਰਾਮਕਲੀ ਰ 299'): the number at either end of a
    top-of-page region the header tests claim. The index's printed pages
    map to scan pages by the offset these vote for.
    """
    from lib.notation_vocab import gurmukhi_digits
    out: dict[int, int] = {}
    repeated = _repeated_tops(layouts, style)
    for lay in layouts:
        page_h = lay.get("page_h") or 0
        regs = lay.get("regions") or []
        if not regs or not page_h:
            continue
        r = regs[0]
        if r["bbox"][1] > 0.1 * page_h or r["role"] not in ("heading", "text", "pageno"):
            continue
        if not (_running_header(r, page_h) or _is_running(r, page_h, repeated)):
            continue
        text = (r.get("text") or "").strip()
        m = re.match(r"^\(?([0-9੦-੯]{1,4})\)?\b", text) or re.search(r"\b\(?([0-9੦-੯]{1,4})\)?\s*$", text)
        if m:
            out[lay["page"]] = int(gurmukhi_digits(m.group(1)))
    return out


def link_pages(layouts: list[dict], style: dict, dropped: list[dict] | None = None) -> list[dict]:
    """
    Regions of consecutive pages, strung into notation spans, one per shabad:
    {"heading": region|None, "heading_page", "headings": [(page, region)], "shabad": [(page, region)],
     "ref": region|None, "sections": [{"label", "kind", "n", "page", "grids", "markers", "taal"}],
     "notes": [(page, region)], "text": [(page, region)], "pages": [..], "extent": {page: bbox},
     "continued": bool, "inherited": bool, "long": bool}

    The shabad is the anchor. A notation is its shabad and everything the
    book prints after it -- grids, taans, tihais, notes, a notation set as
    text -- up to the next shabad, a raag's heading, a page of prose or the
    end of the pages read -- never cut short: a span over more than
    LONG_SPAN_PAGES pages is marked `long` for the reviewer. What has
    no shabad (a preface, an exercise, a raag description) is dropped and
    listed in `dropped`; a notation opened by a heading straight after
    another (the same shabad in a second taal) inherits that one's shabad.
    Where the book sets the shabad after its grids (style.shabad_position
    == "after") the shabad closes the notation instead of opening it.

    A window of pages cuts notations at its edges: a shabad on the last
    page read with nothing after it, and grids on the first page read with
    no shabad before them, are kept as partial notations.
    """
    after = style.get("shabad_position") == "after"
    items = _items(layouts, style)
    if not items:
        return []
    pages_read = sorted({it["page"] for it in items if it["role"] != "break"})
    first_read, last_read = (pages_read[0], pages_read[-1]) if pages_read else (None, None)
    pairing = style.get("pairing") == "section"
    segs = _segments(items, after, pairing)
    if pairing:
        segs = _pair_by_section(segs)
    # the last item of each run of consecutive pages read (a window ends there, or the book)
    tails = {id(it) for k, it in enumerate(items) if it["role"] != "break" and (k + 1 == len(items) or items[k + 1]["role"] == "break")}
    heads = {id(it) for k, it in enumerate(items) if it["role"] != "break" and (k == 0 or items[k - 1]["role"] == "break")}
    out: list[dict] = []
    dropped = dropped if dropped is not None else []
    prev: dict | None = None
    seg_before: dict | None = None            # the segment just before this one, kept or dropped
    for seg in segs:
        roles = {it["role"] for it in seg["items"]}
        pages = sorted({it["page"] for it in seg["items"]})
        has_shabad = "shabad" in roles
        has_content = bool(roles & set(CONTENT_ROLES))
        opener = seg["items"][0]["region"] if seg["items"] and seg["items"][0]["role"] == "heading" else None
        numbered = bool(opener and (opener.get("parsed") or {}).get("number") is not None)
        at_number = any(it["role"] == "note" and _AT_NUMBER.search(it["region"].get("text") or "") for it in seg["items"])
        # a shabad the corpus did not match line for line is still identified when it is of another source, or when the
        # book printed its reference (the resolver finds it by the ang; the next taal's notation inherits the same lines)
        identified = prev is not None and any((it["role"] == "shabad" and it["region"].get("source")) or it["role"] == "ref" for it in prev["items"])
        if not has_shabad and has_content and seg.get("opened_by") == "heading" and not at_number and prev is not None \
                and (prev.get("sid") is not None or identified) and not seg.get("after_break") \
                and any(it["role"] == "shabad" for it in prev["items"]):
            # the same shabad set again in another taal, its text not reprinted
            seg["items"] = [it for it in prev["items"] if it["role"] in ("shabad", "ref")] + seg["items"]
            seg["sid"], seg["inherited"] = prev["sid"], True
            has_shabad = True
        # cut by the edge of the pages read: a shabad on the last page with its grids unread, grids on
        # the first page whose shabad stood on the page before
        shabad_pages = {it["page"] for it in seg["items"] if it["role"] == "shabad"}
        grid_pages = {it["page"] for it in seg["items"] if it["role"] == "grid"}
        edge_shabad = has_shabad and not has_content and ((last_read if not after else first_read) in shabad_pages)
        edge_grid = has_content and not has_shabad and ((first_read if not after else last_read) in grid_pages)
        # a notation the book numbers, with the raag and the taal in its heading, is one
        # even with no shabad text under it ("ਇਹ ਸ਼ਬਦ ਨੰ: ੩ ਤੇ ਲਿਖਿਆ ਹੈ"): the parser
        # borrows the shabad by that note, or reads it off the bol row. A numbered
        # exercise under a taal alone is not.
        parsed = (opener or {}).get("parsed") or {}
        counted = has_content and not has_shabad and numbered and "grid" in roles \
            and ((parsed.get("raag") and parsed.get("taal")) or at_number)
        if not (has_shabad and has_content) and not edge_shabad and not edge_grid and not counted:
            dropped.append({"why": "no-shabad" if not has_shabad else "no-notation", "pages": pages,
                            "roles": sorted(roles), "opened_by": seg.get("opened_by")})
            # a raag's description is where a new section of the book begins: what follows it does not inherit the
            # shabad before it (Guru Angad Dev Sangeet Darpan, pp. 152-177: ten notations of three raags took the
            # Devgandhari pauri when their own, a pauri of Bhai Gurdas, was not read; second cut re-reviewed 6 October 2026)
            prev = seg if has_shabad else (None if seg.get("opened_by") == "raag" else prev)
            seg_before = None                  # a dropped part between two notations: the gap is not filled
            continue
        if edge_grid or (not after and id(seg["items"][0]) in heads and seg["items"][0]["role"] == "shabad"):
            seg["continued"] = True             # grids, or a shabad's text, at the very start of the pages read: it may begin on the page before
        span = _span_from_items(seg)
        last_item = seg["items"][-1]
        if not after and id(last_item) in tails and last_item["role"] in NOTATION_ROLES + ("text", "note") \
                and any(it["role"] == "grid" and it["page"] == last_item["page"] for it in seg["items"]):
            span["continues"] = True            # the last thing read is its grid (or a line under it): the book may go on past the pages read
        # what stands between a notation's last region and the next notation's first on the same
        # page -- a bol row OCR glued to the next verse, a line read as nothing -- is the first's tail:
        # everything between one shabad and the next belongs to the notation
        if out and prev is seg_before and not after:
            last = out[-1]
            for pg, ext in span["extent"].items():
                if pg in last["extent"] and last["extent"][pg][3] < ext[1]:
                    last["extent"][pg][3] = ext[1] - 1
        out.append(span)
        prev = seg
        seg_before = seg
    return out
