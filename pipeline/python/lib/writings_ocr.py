"""
A merged OCR book as pages of paragraphs -- the shape lib/writings_pdf.read_pdf returns.

12_ingest_writings.py reads an author's PDFs into paragraph records without
knowing whether the paragraphs came from a text layer or from OCR: both
readers return {"path", "body_size", "pages": [{"page", "marker", "spread",
"columns", "paragraphs": [{"text", "style", "italic", "x0", "size"}]}]}. This
one reads data/ocr/<book>/merged/NNNN.jsonl (22_ocr_merge.py) and adds, on a
quoted paragraph, the line_ids the merge resolved, so that
13_resolve_citations.py can take them as given instead of guessing from an
English translation.

  marker    the printed page number from the running header ("੨੩" -> "23"),
            playing the part "L127.3" plays for the essays
  quote     a paragraph whose lines are Gurbani (matched or bold with verse
            marks); its text is the corpus's where matched
  footnote  the lines under the rule, one paragraph per numbered note
"""
from __future__ import annotations
import glob
import json
import os
import statistics

from lib.ocr_engines import read_page
from lib.ocr_layout import page_paragraphs, quote_meta
from lib.ocr_zones import FOOTNOTE_START, smooth_hints

CAN_QUOTE = ("gurbani", "gurbani-unmatched")
# the kinds of work whose prose explains the verse before it (a translation
# gives the verse and then its rendering; an essay quotes a verse in passing)
EXPLAINS_KINDS = ("translation", "word-meaning", "commentary")
WIDE_LINE = 0.55        # of the page width: a bold line reaching this far is a sentence with a lead word, not a heading


def page_hints(pages: list[tuple[dict, list]]) -> dict[int, dict]:
    """
    {page: hints} for a book's merged pages, smoothed where the merge did not
    do it (a merge from before smoothing carries no ang_source); a merge
    that did is taken as it is, since running the smoothing again changes
    nothing but would cost the reasons it recorded.
    """
    raw = {pmeta["page"]: dict(pmeta.get("hints") or {}) for pmeta, _ in pages}
    if raw and all("ang_source" in h for h in raw.values()):
        return raw
    return smooth_hints(raw) if raw else {}


def _footnote_paragraphs(lines: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ln in lines:
        text = ln.get("text", "").strip()
        if not text:
            continue
        if out and not FOOTNOTE_START.match(text):
            out[-1]["text"] += " " + text
            out[-1]["lines"].append(ln)
        else:
            out.append({"text": text, "style": "footnote", "italic": False, "x0": ln["bbox"][0],
                        "size": ln["bbox"][3] - ln["bbox"][1], "lines": [ln]})
    return out


def _explains(verse_lines: list[dict], pair: int) -> dict:
    """What a prose paragraph explains: the verse lines beside it, by their corpus match if they have one."""
    qm = quote_meta({"lines": verse_lines})
    if not qm:
        return {"pair": pair, "unmatched": True}
    return {"pair": pair, "shabad_id": qm["shabad_id"], "line_from": qm["line_from"], "line_to": qm["line_to"],
            "ang": qm["ang"], "source": qm["source"], "score": qm["match_score"], "method": qm["match_method"]}


def within_angs(verse: dict | None, angs: list | None) -> bool:
    """
    Whether a verse can be one the work explains: inside the manifest's
    `angs` when the work states them and the verse was matched. A commentary
    on angs 1-53 quotes a line from ang 1043 to illustrate a point, and the
    prose after it is about the pauri, not about that line: 834 of 1,774
    explains links on the Santhya pointed outside its volume before this.
    """
    if not angs or not verse or verse.get("ang") is None:
        return True
    lo, hi = angs[0], angs[-1]
    return lo <= verse["ang"] <= hi


def read_ocr_book(book_dir: str, layout: str = "auto", kind: str = "essay", angs: list | None = None) -> dict:
    """
    @param layout  how a two-column page is read (lib/ocr_layout.page_paragraphs):
                   "auto" pairs a verse column with the explanation beside it
    @param kind    what the work is to the scripture (the manifest's `kind`): for a
                   translation, word-meaning or commentary the prose that FOLLOWS a
                   verse on a single-column page explains it too; an essay's does not
    @param angs    the angs the work covers (the manifest's `angs`; default: pages.json's):
                   on a single-column page only a verse inside them is explained by the
                   prose after it; a verse from elsewhere is a quotation
    """
    with open(os.path.join(book_dir, "pages.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    angs = angs if angs is not None else meta.get("angs")
    pages = []
    sizes: list[float] = []
    read = [read_page(path) for path in sorted(glob.glob(os.path.join(book_dir, "merged", "*.jsonl")))]
    hints_of = page_hints(read)
    follows = kind in EXPLAINS_KINDS
    last_verse: dict | None = None            # the verse chunk an explanation may run on from, across a page
    for pmeta, lines in read:
        body = [ln for ln in lines if ln.get("zone") == "body"]
        foot = [ln for ln in lines if ln.get("zone") == "footnote"]
        columns = pmeta.get("columns") or None
        paras = page_paragraphs(body, pmeta["page_h"], columns, layout=layout, typical_h=pmeta.get("body_h"))
        out = []
        page_layout = "single" if not columns else ("paired" if any(p.get("pair") for p in paras) else "columns")
        open_pair: int | None = None          # single-column alternation: the verse a following paragraph explains
        next_pair = max([p.get("pair", 0) for p in paras] + [0])
        for p in paras:
            kinds = [ln.get("kind") for ln in p.get("lines", [])]
            # a quotation is verse lines, matched or bold with a verse mark; a
            # short bold line among them (kind heading) is the shabad's raag
            # title or a wrapped fragment, part of the quotation, not a reason
            # to read the whole block as prose
            quote = bool(p.get("verse")) or (bool(kinds) and all(k in CAN_QUOTE or k == "heading" for k in kinds)
                                             and any(k in CAN_QUOTE for k in kinds))
            if quote:
                style = "quote"
            elif p["style"] == "quote":
                # bold prose is not a quotation: a section title ("( ਪਉੜੀ ੩ )")
                # is a heading, a sentence with a bold lead word ("ਪ੍ਰਾਕਥਨ- ਜਿਸ
                # ਦੇ ਹੁਕਮ ਦਾ ...") is body. A heading is short and does not
                # reach across the page; the lead-word line does
                xs = [ln["bbox"] for ln in p.get("lines", []) if ln.get("bbox")]
                reach = (max(b[2] for b in xs) - min(b[0] for b in xs)) / float(pmeta.get("page_w") or 1) if xs else 0.0
                style = "heading" if len(p["text"]) < 80 and reach < WIDE_LINE else "body"
            else:
                style = p["style"]
            rec = {"text": p["text"], "style": style, "italic": quote, "x0": p["x0"], "size": p["size"],
                   "kinds": kinds}
            if quote:
                rec.update(quote_meta(p))
            if p.get("pair"):
                rec["pair"] = p["pair"]
            if quote:
                last_verse = _explains(p.get("lines", []), p.get("pair", 0))
                if (not p.get("pair") and style == "quote" and follows and page_layout != "paired"
                        and within_angs(last_verse, angs)):
                    next_pair += 1
                    open_pair = next_pair
                    rec["pair"] = open_pair
                    last_verse["pair"] = open_pair
            elif "explains_lines" in p:
                # a paired band: the explanation beside its verse, whatever the work is
                if p["explains_lines"]:
                    rec["explains"] = [_explains(p["explains_lines"], p["pair"])]
                elif last_verse:
                    rec["explains"] = [{**last_verse, "pair": p["pair"], "continues": True}]
            elif style == "body" and open_pair and last_verse:
                # single column: the prose after a verse explains it, for a work of that kind
                rec["pair"] = open_pair
                rec["explains"] = [{**last_verse, "pair": open_pair}]
            elif style == "heading":
                open_pair = None
            if quote and not within_angs(last_verse, angs):
                last_verse, open_pair = None, None          # a quotation from elsewhere explains nothing after it
            if any(ln.get("route") for ln in p.get("lines", [])):
                rec["routed"] = True
            agreements = [ln["agreement"] for ln in p.get("lines", []) if ln.get("agreement") is not None]
            if agreements:
                rec["agreement"] = round(min(agreements), 3)
            out.append(rec)
            sizes.append(p["size"])
        for fp in _footnote_paragraphs(foot):
            out.append({k: v for k, v in fp.items() if k != "lines"})
        hints = hints_of.get(pmeta["page"]) or {}
        marker = str(hints["book_page"]) if hints.get("book_page") else None
        pages.append({"page": pmeta["page"], "marker": marker, "spread": False,
                      "columns": len(columns or []), "paragraphs": out, "layout": page_layout,
                      "ang_from": hints.get("ang_from"), "ang_to": hints.get("ang_to"),
                      "ang_source": hints.get("ang_source")})
    return {"path": meta.get("pdf"), "body_size": statistics.median(sizes) if sizes else 10.0, "pages": pages}
