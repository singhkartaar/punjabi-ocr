"""
Merge every engine's reading of a book into one text per page.

  22_ocr_merge.py --book santhya-vol-1
  22_ocr_merge.py --book santhya-vol-1 --engines tesseract-pan,tesseract-gurmukhi,vision --pivot tesseract-pan
  22_ocr_merge.py --book ten-masters --no-correct

Per page, on the pivot engine's lines (the others are aligned to them by box):

  zones      header / footnote / stamp / page number set aside (lib/ocr_zones)
  match      every Gurmukhi body line is looked for in the corpus, using every
             engine's reading of it; a line that is found becomes the corpus
             text, carries its line_ids, and is protected from everything below
  vote       the other lines are voted grapheme by grapheme across engines
             (lib/ocr_vote); the agreement is recorded
  lexicon    the voted text's out-of-vocabulary rate (lib/ocr_lexicon)
  correct    unknown words one known confusion away from a known word are
             corrected (lib/ocr_correct); --no-correct turns it off, and the
             evaluation decides whether it stays on for a book
  bold       stroke width splits bold from regular on pages that carry both
             (lib/ocr_layout); bold + verse marks = Gurbani the corpus did not
             hold (a Dasam Granth or Bhai Gurdas line before those sources exist)
  route      lines and pages that want a paid arbiter (lib/ocr_route)

Writes data/ocr/<book>/merged/NNNN.jsonl, merged/route.json and
data/raw/ocr-report-<book>.json. Nothing here is sent anywhere. `--out-name
merged-<x>` writes merged-<x>/ and ocr-report-<book>-<x>.json instead, so a
second configuration can be scored beside the first (24_ocr_eval.py --merged).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ocr_correct import Confusions, Corrector, shared_misread
from lib.ocr_coverage import crop_box, is_text_reading, typical_height, uncovered_regions, MAX_REGIONS
# the page-segmentation mode to fall back to when a crop reads empty: 7 (one line) -> 13 (raw line),
# 6 (a block) -> 4 (a column of lines of varying size); both skip Tesseract's own layout analysis
RAW_PSM = {7: 13, 6: 4}
from lib.ocr_engines import load_engine, read_page, tess_lang_of, write_page
from lib.ocr_layout import bold_split, stroke_width
from lib.ocr_lexicon import Lexicon
from lib.ocr_match import CorpusIndex, align_stream, match_text_all
from lib.ocr_pages import parse_pages
from lib.ocr_route import needs_arbiter, page_needs_arbiter
from lib.ocr_text import GURMUKHI, LATIN, key_positions, normalise, script_of, splice, words
from lib.ocr_vote import H_OVERLAP, align_boxes, box_overlap, is_block, joined_text, segment_block, vote
from lib.ocr_zones import (classify_zones, footnote_rule_y, header_of, is_stamp, page_columns_with_source,
                           smooth_hints, vertical_rule)
from lib.paths import CORPUS_DB, GRANTHS_DB, MAHANKOSH_DB, OCR_DIR, ROOT, SCRIPTURES_FULL_DB

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CONFUSIONS = os.path.join(OCR_DIR, "confusions.json")
RAW = os.path.join(ROOT, "data", "raw")
VERSE_MARK = "॥"
HEADING_MAX_WORDS = 6
# a plain prose line must be mostly verse for a contained match to count as a quotation
PROSE_COVERAGE = 0.6
_RAAG_HEADING = re.compile(r"ਮਹਲਾ\s*[੧-੯ਮ]|ਮਃ\s*[੧-੯]")   # ਮਹਲਾ ੧ / ਮਃ ੧


def load_index(corpus: str, granths: str, scriptures: str | None = None) -> CorpusIndex | None:
    """
    The Granth's lines, then the other scriptures'. The full store
    (data/scriptures-full.sqlite: every source, selected or not) is preferred to granths.sqlite,
    which predates it and knows only D and B. The window a page header gives is
    the Granth's (Merger.sources); the other scriptures are found by the global
    pass, at its stricter threshold and under the same margin rule.
    """
    rows = []
    if corpus and os.path.exists(corpus) and os.path.getsize(corpus) > 0:
        con = sqlite3.connect(corpus)
        # headings ("ਮਾਰੂ ਮਹਲਾ ੫ ॥") are bibliographic and would match the
        # citations the books print after a quote
        rows += [(*r, "G") for r in con.execute(
            "SELECT line_id, shabad_id, ang, gurmukhi_uni FROM lines WHERE kind != 'heading' ORDER BY line_id")]
        con.close()
    if scriptures and os.path.exists(scriptures) and os.path.getsize(scriptures) > 0:
        rows += CorpusIndex.scripture_rows(scriptures)
    elif granths and os.path.exists(granths) and os.path.getsize(granths) > 0:
        con = sqlite3.connect(granths)
        try:
            rows += list(con.execute("SELECT line_id, shabad_id, ang, text, source FROM granth_lines"))
        except sqlite3.OperationalError:
            pass
        con.close()
    return CorpusIndex(rows) if rows else None


MIN_VOTER_ACC = 0.90      # an engine measured below the project's bar does not vote


def measured_accuracy(book: str) -> dict[str, float]:
    """Word accuracy per engine from the book's evaluation (24_ocr_eval.py), if any."""
    path = os.path.join(RAW, "ocr-eval-%s.json" % book)
    acc: dict[str, float] = {}
    if not os.path.exists(path):
        print("no %s: every engine votes with the same weight (23_ocr_gt.py + 24_ocr_eval.py measure them)"
              % os.path.relpath(path, ROOT))
        return acc
    with open(path, encoding="utf-8") as fh:
        for r in json.load(fh).get("engines", []):
            acc[r["engine"]] = float(r.get("word_acc") or 0.0)
    return acc


def auto_weights(acc: dict[str, float], engines: list[str]) -> dict[str, float]:
    """
    Vote weights from the measured word accuracy, to the fourth power, so a
    99% engine (0.96) is not outvoted by a 77% one (0.35) and an 88% one
    (0.60) together. Measured on Ten Masters: with equal weights the pdftext
    and Surya readings pulled a 99.1% Tesseract pivot down to 97.3%. An
    engine without a measurement votes at 0.5.
    """
    return {e: round(acc[e] ** 4, 3) if e in acc else 0.5 for e in engines}


def choose_voters(engines: list[str], acc: dict[str, float], pivot: str) -> tuple[list[str], list[str]]:
    """
    The engines that vote, and the ones set aside: a measured engine below
    MIN_VOTER_ACC does not vote, whatever its weight. Measured on the Santhya's
    15 ground-truth pages: the three Tesseract variants merge to 94.9%; adding
    Surya (92.1%) leaves 94.9%; adding dots.ocr (88.3%) and IndicOCR (83.1%)
    drops the merge to 90.6%, and steeper weights (accuracy^8) change nothing --
    a block-level engine's damage is in how its lines align with the pivot's,
    not in how often it wins a vote. The pivot always votes; an unmeasured
    engine votes at 0.5 as before. --engines names voters explicitly and
    bypasses this.

    The bar is for an engine of another family. A Tesseract variant beside a
    Tesseract pivot reads the same lines the same way, and always votes: on
    the Santhya's harder scans (vols. 2, 4, 5, 7, measured 2026-10-01 on 30
    ground-truth lines each) both variants measure 0.78-0.91, so the bar left
    the pivot alone -- no vote, no book vocabulary learnt from agreement, not
    one correction. With both voting the merge read 0.817 -> 0.830, 0.820 ->
    0.825, 0.913 -> 0.923 and 0.856 -> 0.854, learnt ~5,000 book words each,
    and halved the unknown words (oov_mean 0.25 -> 0.14 on vol. 4).
    """
    keep, drop = [], []
    for e in engines:
        if e != pivot and e in acc and acc[e] < MIN_VOTER_ACC and not same_family(e, pivot):
            drop.append(e)
        else:
            keep.append(e)
    return keep, drop


def same_family(a: str, b: str) -> bool:
    """Two variants of one engine (tesseract-pan, tesseract-gurmukhi): the same layout, line for line."""
    return a.split("-")[0] == b.split("-")[0]


def default_engines(book_dir: str) -> list[str]:
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(book_dir, "ocr", "*")) if os.path.isdir(d))


def default_pivot(engines: list[str], lang: str) -> str:
    for cand in (["tesseract-pan", "tesseract", "tesseract-gurmukhi"] if lang == "pa" else ["tesseract"]):
        if cand in engines:
            return cand
    return engines[0]


_DOUBLE_DANDA = re.compile("(॥)(\\s*[॥।|])+")


def replaced_text(text: str, matches: list[dict]) -> str:
    """The reading with each matched span replaced by the corpus line."""
    key, spans = key_positions(text)
    reps = []
    for m in matches:
        s, e = m["span"] if m["span"] else (0, len(key))
        s, e = max(0, min(s, len(spans) - 1)), max(1, min(e, len(spans)))
        reps.append((spans[s][0], spans[e - 1][1], m["text"]))
    out = splice(text, reps)
    # the corpus line ends in its own danda; the reading's danda sat outside
    # the matched span and would follow it
    return _DOUBLE_DANDA.sub("\\1", out)


_SEAM_JUNK = re.compile("^[\\s।॥|.,;:'\"()\\-]+")


def tidy_seams(text: str, corpus_texts: list[str]) -> str:
    """
    Residue at the joins of a spliced line: leading dandas and punctuation,
    and a word before the corpus text that merely repeats its first word
    (the alignment put the span boundary one word late).
    """
    text = _SEAM_JUNK.sub("", text)
    for ct in corpus_texts:
        first = ct.split(" ")[0]
        pos = text.find(ct)
        if pos <= 0 or not first:
            continue
        before = text[:pos].rstrip()
        if before.endswith(first) and (len(before) == len(first) or before[-len(first) - 1] == " "):
            text = before[:-len(first)].rstrip() + " " + text[pos:]
            text = text.strip()
    return re.sub(r"\s+", " ", text).strip()


def adopt_orphans(pivot_body: list[dict], others: dict[str, list[dict]], typical_h: float) -> list[dict]:
    """
    Lines another engine read that no pivot line covers -- a bold verse line
    Tesseract's segmenter skipped -- adopted into the body so the page keeps
    them. The adopting engine's text becomes the line's pivot reading.

    Covered means overlapped in BOTH directions. Judged on height alone, a
    left-column line the pivot skipped was "covered" by the right-column line
    printed beside it, and on a two-column page nothing was ever adopted.
    """
    adopted: list[dict] = []
    for engine, lines in others.items():
        for ln in lines:
            if not ln.get("text", "").strip() or (ln["bbox"][3] - ln["bbox"][1]) > 2.5 * typical_h:
                continue
            covered = any(v >= 0.4 and h >= H_OVERLAP
                          for v, h in (box_overlap(ln["bbox"], p["bbox"]) for p in pivot_body + adopted))
            if not covered:
                adopted.append({**ln, "words": ln.get("words") or [], "adopted_from": engine})
    return adopted


def disputed_words(pivot_text: str, other_texts: list[str]) -> set[str]:
    """Words of the pivot's reading that at least one other engine read differently."""
    if not other_texts:
        return set()
    mine = set(words(pivot_text))
    out = set()
    for t in other_texts:
        theirs = set(words(t))
        out |= mine - theirs
    return out


def split_at_gutter(lines: list[dict], at: int, tol: int = 20, span: tuple[int, int] | None = None) -> list[dict]:
    """
    A body line the engine read across both columns, cut at the gutter by its
    word boxes. Tesseract's page segmentation usually keeps the columns apart;
    where a hairline rule fooled it, "ਖਲਾਵੈ ਕਵਣੁ ... | ਚੁਗਾਉਂਦਾ ਹੈ?" is two lines.

    `span` is the rows the gutter runs down. A rule beside one band of verse
    does not make the prose above it two columns: a line outside the span is
    left whole.
    """
    out = []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        ws = ln.get("words") or []
        inside = span is None or (y1 > span[0] - tol and y0 < span[1] + tol)
        if ln.get("zone") not in ("body", "footnote") or not ws or not inside or not (x0 < at - tol and x1 > at + tol):
            out.append(ln)
            continue
        # the rule itself, read as a word ("|", "_", "।") at the gutter, is nobody's
        ws = [w for w in ws if not (abs((w["bbox"][0] + w["bbox"][2]) / 2.0 - at) <= tol
                                    and not any(c.isalpha() for c in w.get("text", "")))]
        left = [w for w in ws if (w["bbox"][0] + w["bbox"][2]) / 2.0 < at]
        right = [w for w in ws if (w["bbox"][0] + w["bbox"][2]) / 2.0 >= at]
        if not left or not right:
            out.append(ln)
            continue
        for part in (left, right):
            out.append({**ln, "bbox": [min(w["bbox"][0] for w in part), min(w["bbox"][1] for w in part),
                                       max(w["bbox"][2] for w in part), max(w["bbox"][3] for w in part)],
                        "text": " ".join(w["text"] for w in part), "words": part, "split": True})
    out.sort(key=lambda l: (l["bbox"][1] // 10, l["bbox"][0]))
    for i, ln in enumerate(out, start=1):
        ln["n"] = i
    return out


SPECK_H = 0.3   # of the line height: a split piece shorter than this is a fragment of the rule, not a line
SPECK_W = 1.0   # of the line height: a piece narrower than this AND under half a line tall is one too


def drop_specks(lines: list[dict], typical_h: float, lang: str) -> int:
    """
    Split pieces that are fragments of the rule -- a few pixels tall, with no
    letter in them ("=", "---------"), or a speck narrower than a line is
    tall and under half a line high ("HL ਮੁਲ" in 67 by 26 px) -- become zone
    "speck" and are dropped. Tesseract reads a hairline rule as a word now and then, and the
    gutter split leaves that word alone in a piece of its own; left in the
    body it is a line of prose to the reader, and on page 70 of the Santhya
    one "=" beside an arth made a full-width row that broke the verse-and-arth
    band in two. Returns how many were dropped.
    """
    n = 0
    for ln in lines:
        if not ln.get("split") or ln.get("zone") not in ("body", "footnote"):
            continue
        h, w = ln["bbox"][3] - ln["bbox"][1], ln["bbox"][2] - ln["bbox"][0]
        if (h < SPECK_H * typical_h or is_text_reading(ln.get("text", ""), lang) is not None
                or (w < SPECK_W * typical_h and h < 0.5 * typical_h)):
            ln["zone"] = "speck"
            n += 1
    return n


class Merger:
    def __init__(self, book_dir: str, meta: dict, engines: list[str], pivot: str, weights: dict,
                 index: CorpusIndex | None, lexicon: Lexicon | None, corrector: Corrector | None,
                 coverage: dict | None = None, correct_agreed: bool = False):
        self.book_dir, self.meta = book_dir, meta
        self.correct_agreed = correct_agreed
        self.engines, self.pivot, self.weights = engines, pivot, weights
        self.index, self.lexicon, self.corrector = index, lexicon, corrector
        self.lang = meta.get("language", "en")
        self.sources = ["G"] + ([meta["scripture"]] if meta.get("scripture", "G") != "G" else [])
        # {"psm", "max", "min_agreement", "route"} when the coverage pass is on
        self.coverage = coverage
        self.recognisers: dict[str, object] = {}
        # the book's own header shape, when the manifest gave one
        self.pattern = re.compile(meta["header_pattern"]) if meta.get("header_pattern") else None
        self.hints: dict[int, dict] = {}
        self.hints_raw: dict[int, dict] = {}
        # a notation book: the grid's rules would pass for the footnote rule,
        # its column gaps for a gutter, and its swara rows are all out of
        # vocabulary, so those three stay off; the corpus matching of the
        # shabad text is what names the shabad (29_notation_parse.py)
        self.notation = meta.get("kind") == "notation"
        if self.notation:
            self.corrector = None

    def read(self, engine: str, page: int):
        p = os.path.join(self.book_dir, "ocr", engine, "%04d.jsonl" % page)
        return read_page(p) if os.path.exists(p) else (None, None)

    def header_pass(self, pages: list[dict]) -> None:
        """
        Every page's header hints, read first and smoothed together
        (lib/ocr_zones.smooth_hints), so a misread ang on one page is put
        right by its neighbours before it becomes that page's matching
        window. One small file per page; nothing is rendered.
        """
        raw: dict[int, dict] = {}
        for rec in pages:
            pmeta, lines = self.read(self.pivot, rec["page"])
            if pmeta is None:
                continue
            zoned = classify_zones(lines, pmeta["page_w"], pmeta["page_h"], rec.get("stamps"))
            raw[rec["page"]] = header_of(zoned, self.pattern)
        self.hints_raw = raw
        self.hints = smooth_hints(raw)

    def vocab_pass(self, pages: list[dict]) -> int:
        """
        The book's own vocabulary (Lexicon.add_book_vocab): a word two engines
        both read on one page, on MIN_SEEN pages, is a word of this book, so the
        author's spelling is not "corrected" to a dictionary's. Every page, as
        the header pass, whatever --pages asks for. On a grey scan the engines
        disagree on a misread conjunct (pan drops the subjoined ra, gurmukhi
        writes an aunkar), so the misreadings do not get in.
        """
        if self.lexicon is None:
            return 0
        readings = []
        for rec in pages:
            texts = []
            for e in self.engines:
                pmeta, lines = self.read(e, rec["page"])
                if pmeta is not None:
                    texts.append(" ".join(ln.get("text", "") for ln in lines))
            readings.append(texts)
        lex = self.lexicon
        # a word both engines misread the same way is refused (shared_misread):
        # admitted, ਗੰਥ tied with ਗ੍ਰੰਥ over Sant Attar Singh vol. 1's 466 pages
        # and 66 ਗੁੰਥ stayed
        seen = {w for texts in readings for t in texts for w in words(t)}
        return lex.add_book_vocab(readings, refuse=lambda w: shared_misread(w, lex.knows, seen))

    def recogniser(self, engine: str):
        """The Tesseract variant behind an engine key, built once; None for any other engine."""
        if engine not in self.recognisers:
            is_tess = engine == "tesseract" or engine.startswith("tesseract-")
            self.recognisers[engine] = load_engine("tesseract", tess_lang=tess_lang_of(engine)) if is_tess else None
        return self.recognisers[engine]

    def recover_regions(self, img, regions: list[dict], columns: list, typical_h: float, page_w: int, page_h: int,
                        rule_y: int | None) -> tuple[list[dict], dict[str, list[dict]]]:
        """
        The candidate regions read as lines: (the pivot's lines, each with a
        `recovered` block saying which region, psm, crop and confidence; the
        other Tesseract voters' readings of the same crops, keyed by engine,
        for the vote). A region whose reading is empty or not text is marked
        rejected in place, with the reason. Only Tesseract reads crops: the
        others read whole pages, and a paid one bills per image.
        """
        pivot = self.recogniser(self.pivot)
        if pivot is None:
            return [], {}
        others = {e: self.recogniser(e) for e in self.engines if e != self.pivot}
        others = {e: r for e, r in others.items() if r is not None}
        lines: list[dict] = []
        alts: dict[str, list[dict]] = {e: [] for e in others}
        zone_of = lambda y: "footnote" if rule_y is not None and y > rule_y else "body"
        for k, region in enumerate(regions):
            if region["status"] != "candidate":
                continue
            band = tuple(columns[region["col"]]) if columns else (0, page_w)
            crop = crop_box(region, band, typical_h, page_w, page_h)
            psm = self.coverage["psm"] if region["h_rel"] <= 1.6 else 6
            read = self.read_crop(pivot, img, crop, psm)
            if read is None:
                region.update(status="rejected", why="engine-error")
                continue
            text = " ".join(ln["text"] for ln in read).strip()
            why = is_text_reading(text, self.lang)
            if why in ("empty", "no-letters") and psm in RAW_PSM:
                # Tesseract's segmenter refused the crop for the same reason the
                # page layout skipped the line (a grey band behind the text, a
                # line it takes for a picture): the raw modes bypass the
                # segmenter and read the crop as the line or block it is.
                # Measured: 719 crops on the Santhya and 61 on a 120 dpi scan
                # came back empty at psm 7; every one sampled was a real line,
                # and psm 13 read them all
                raw = RAW_PSM[psm]
                read2 = self.read_crop(pivot, img, crop, raw) or []
                text2 = " ".join(ln["text"] for ln in read2).strip()
                if is_text_reading(text2, self.lang) is None:
                    read, text, why, psm = read2, text2, None, raw
                    region["raw"] = True
            if why is None and self.coverage["min_agreement"] > 0 and others:
                # the other variants read the crop too; a reading nobody agrees
                # with is refused when the knob is set (off by default: two
                # Tesseract models share too much to be independent witnesses)
                from rapidfuzz import fuzz
                theirs = [" ".join(ln["text"] for ln in self.read_crop(r, img, crop, psm) or []) for r in others.values()]
                if theirs and max(fuzz.ratio(normalise(text), normalise(t)) for t in theirs) / 100.0 < self.coverage["min_agreement"]:
                    why = "low-agreement"
            if why is not None:
                region.update(status="rejected", why=why)
                continue
            conf = round(statistics.mean(ln["conf"] for ln in read if ln.get("conf") is not None), 3) \
                if any(ln.get("conf") is not None for ln in read) else None
            for ln in read:
                # Tesseract's line box for a one-line crop is the crop, half a
                # line of padding included -- and in raw mode its word boxes
                # are as tall as the crop too: 1.4 to 2.6 line heights, which
                # the reader took for a heading and a row for two. Across, the
                # line is where its words are; up and down, where the ink was
                # (the region), with a quarter line for the matras and
                # descenders the ink band may have cut off.
                ws = ln.get("words") or []
                pad = int(round(0.25 * typical_h))
                rx0, ry0, rx1, ry1 = region["bbox"]
                ln["bbox"] = [min(w["bbox"][0] for w in ws) if ws else rx0, max(crop[1], ry0 - pad),
                              max(w["bbox"][2] for w in ws) if ws else rx1, min(crop[3], ry1 + pad)]
                ln.update({"zone": zone_of(ln["bbox"][1]),
                           "recovered": {"region": k, "psm": psm, "engine": self.pivot, "crop": crop, "conf": conf}})
                lines.append(ln)
            region["status"] = "recovered"
            for e, r in others.items():
                alts[e].extend(self.read_crop(r, img, crop, psm) or [])
        return lines, alts

    def read_crop(self, engine, img, crop, psm: int) -> list[dict] | None:
        """
        One crop read by one Tesseract variant, or None when Tesseract itself
        died on it. Measured: on page 371 of Sant Attar Singh vol. 1 a crop
        killed Tesseract with a floating-point exception (signal 8) and took
        the whole book's merge down with it; such a region is now refused as
        engine-error and counted, and the page goes on.
        """
        try:
            return engine.recognise_region(img, crop, self.lang, psm)
        except Exception as err:                              # noqa: BLE001
            if type(err).__name__ != "TesseractError":
                raise
            print("  tesseract failed on a crop %s at psm %d: %s" % (crop, psm, err), flush=True)
            return None

    def match(self, texts: list[str], hints: dict, quotable: bool = True) -> tuple[list[dict], str | None]:
        """
        The best set of corpus matches over every reading of a line.

        quotable: the line is bold or carries a verse mark. A plain prose
        line is only a quotation when the match covers most of it: a
        sentence that mentions a phrase from a verse is not quoting the verse.
        """
        best, best_text, best_score = [], None, 0.0
        for t in texts:
            if not t or not GURMUKHI.search(t):
                continue
            key_len = len(key_positions(t)[0]) or 1
            for src in self.sources:
                found = match_text_all(t, self.index, hints, source=src)
                if found and not quotable:
                    covered = sum((m["span"][1] - m["span"][0]) if m["span"] else key_len for m in found)
                    if covered < PROSE_COVERAGE * key_len:
                        found = []
                score = sum(m["score"] for m in found)
                if found and score > best_score:
                    best, best_text, best_score = found, t, score
        return best, best_text

    def stream_matches(self, body: list[dict], bold: list[bool], cols: list[int], hints: dict) -> dict[int, dict]:
        """
        Runs of consecutive bold-or-verse lines in one column, matched as one
        stream (lib/ocr_match.align_stream) BEFORE the per-line pass: a corpus
        line that wraps over several OCR lines lands on the first of them and
        the others keep only what lies outside the match. Matching the
        fragments one by one instead would give the first fragment the whole
        corpus line and leave the rest to duplicate it.

        @returns {body index: {"text", "matches" | "merged_into"}}
        """
        runs: list[list[int]] = []
        # runs are built per column: on a two-column page the commentary
        # lines of the right column interleave with the verse of the left
        for col in sorted(set(cols)):
            last = None                                   # the previous line of this column, eligible or not
            for i, ln in enumerate(body):
                if cols[i] != col:
                    continue
                eligible = GURMUKHI.search(ln.get("text", "")) and (bold[i] or VERSE_MARK in ln["text"])
                if eligible:
                    # a run continues only when the previous line of the column
                    # is the run's last member: a commentary line between breaks it
                    if runs and last is not None and runs[-1][-1] == last:
                        runs[-1].append(i)
                    else:
                        runs.append([i])
                last = i
        out: dict[int, dict] = {}
        for run in runs:
            if len(run) < 2:
                continue
            keyed = [key_positions(body[i]["text"]) for i in run]
            found = []
            for src in self.sources:
                found = align_stream([k for k, _ in keyed], self.index, hints, source=src)
                if found:
                    break
            for m in found:
                anchor = None
                for j, s0, e0 in m["lines"]:
                    i = run[j]
                    key, spans = keyed[j]
                    if not spans or e0 <= s0:
                        continue
                    a = spans[max(0, min(s0, len(spans) - 1))][0]
                    b = spans[max(0, min(e0, len(spans)) - 1)][1]
                    rec = out.setdefault(i, {"text": body[i]["text"], "matches": [], "_edits": []})
                    if anchor is None:
                        rec["_edits"].append((a, b, m["text"]))
                        rec["matches"].append({k: v for k, v in m.items() if k != "lines"} | {"stream": True})
                        anchor = body[i]["n"]
                    else:
                        rec["_edits"].append((a, b, ""))
                        rec["merged_into"] = anchor
        # every edit of a line applied at once on the ORIGINAL positions, then
        # the seams tidied: a wrapped verse leaves a stray danda or the first
        # word of the next line's verse behind at the join
        for i, rec in out.items():
            text = _DOUBLE_DANDA.sub("\\1", splice(body[i]["text"], rec.pop("_edits")))
            rec["text"] = tidy_seams(text, [m["text"] for m in rec["matches"]])
            if not rec["matches"] and not GURMUKHI.search(rec["text"]):
                rec["text"] = ""                           # a continuation line with nothing of its own left
        return out

    def merge_page(self, rec: dict) -> tuple[dict, list[dict]] | None:
        import cv2
        page = rec["page"]
        pmeta, pivot_lines = self.read(self.pivot, page)
        if pmeta is None:
            return None
        img = cv2.imread(os.path.join(self.book_dir, "pages", rec["file"]), cv2.IMREAD_GRAYSCALE)
        rule = footnote_rule_y(img) if img is not None and not self.notation else None
        lines = classify_zones(pivot_lines, pmeta["page_w"], pmeta["page_h"], rec.get("stamps"), rule)
        if self.notation:
            # the shabad's first line, or a heading, may sit in the header band
            for ln in lines:
                t = ln.get("text") or ""
                if ln["zone"] == "header" and (VERSE_MARK in t or len(words(t)) >= 6):
                    ln["zone"] = "body"
        raw_hints = header_of(lines, self.pattern)
        hints = dict(self.hints.get(page) or raw_hints)
        if not hints.get("ang_from") and self.meta.get("angs"):
            # nothing on the page or its neighbours: the manifest's range is
            # the window, wide as it is
            angs = self.meta["angs"]
            hints.update(ang_from=angs[0], ang_to=angs[-1], ang_source="manifest")
        wordboxes = [w for ln in lines for w in ln.get("words", [])] or lines
        vrule = vertical_rule(img, pmeta["page_w"]) if img is not None else None
        rule_x = vrule["x"] if vrule else None
        # a notation book: its grid's column gaps are no gutter
        columns, columns_source = ([], "notation") if self.notation else page_columns_with_source(wordboxes, pmeta["page_w"], img, lines, rule_x)
        if columns:
            lines = split_at_gutter(lines, columns[0][1],
                                    span=(vrule["y0"], vrule["y1"]) if columns_source == "rule" else None)
        body = [ln for ln in lines if ln["zone"] in ("body", "footnote") and ln.get("text", "").strip()]
        typical_h = typical_height(body, pmeta["page_h"])
        if drop_specks(lines, typical_h, self.lang):
            body = [ln for ln in body if ln["zone"] in ("body", "footnote")]

        # other engines, blocks cut at the pivot's lines, then paired by box
        expanded_by_engine: dict[str, list[dict]] = {}
        for e in self.engines:
            if e == self.pivot:
                continue
            emeta, elines = self.read(e, page)
            if emeta is None:
                continue
            expanded: list[dict] = []
            for ln in elines:
                if is_block(ln, typical_h):
                    expanded.extend(segment_block(ln, body))
                else:
                    expanded.append(ln)
            if columns:
                # the other engine read the same rows across the same gutter:
                # its lines are cut where the pivot's were, or a whole row's
                # text would be voted onto each half of it
                expanded = split_at_gutter(classify_zones(expanded, pmeta["page_w"], pmeta["page_h"], rec.get("stamps"), rule),
                                           columns[0][1],
                                           span=(vrule["y0"], vrule["y1"]) if columns_source == "rule" else None)
            expanded_by_engine[e] = expanded
        # a line the pivot's segmenter skipped but another engine read is
        # adopted; it then takes part in the vote like any pivot line
        # coverage is judged against EVERY pivot line, headers and stamps
        # included, or the other engines' header readings would be adopted
        orphans = adopt_orphans(lines, expanded_by_engine, typical_h)
        if orphans:
            for o in orphans:
                o["zone"] = "body"
            body = sorted(body + orphans, key=lambda l: (l["bbox"][1], l["bbox"][0]))
            lines = sorted(lines + orphans, key=lambda l: (l["bbox"][1], l["bbox"][0]))
            for k, ln in enumerate(lines, start=1):
                ln["n"] = k
        # the ink no line covers, read from crops and added as lines
        # (lib/ocr_coverage): every region, taken or refused, goes to _meta
        cov: dict = {"on": False}
        if self.coverage and img is not None:
            t_cov = time.time()
            found = uncovered_regions(img, lines, columns, typical_h, pmeta["page_w"], pmeta["page_h"],
                                      rule_x=rule_x, rule_y=rule, stamps=rec.get("stamps"),
                                      max_regions=self.coverage["max"])
            recovered, alts = self.recover_regions(img, found["regions"], columns, typical_h,
                                                   pmeta["page_w"], pmeta["page_h"], rule)
            if recovered:
                body = sorted(body + recovered, key=lambda l: (l["bbox"][1], l["bbox"][0]))
                lines = sorted(lines + recovered, key=lambda l: (l["bbox"][1], l["bbox"][0]))
                for k, ln in enumerate(lines, start=1):
                    ln["n"] = k
                for e, extra in alts.items():
                    expanded_by_engine.setdefault(e, []).extend(extra)
            rejected = Counter(r["why"] for r in found["regions"] if r["status"] == "rejected")
            cov = {"on": True, "psm": self.coverage["psm"], "residual_share": found["residual_share"],
                   "regions": found["regions"], "recovered": len(recovered),
                   "rejected": dict(rejected), "capped": found["capped"],
                   "uncovered_after": sum(1 for r in found["regions"] if r["status"] != "recovered"
                                          and r["why"] not in ("short", "narrow", "sparse", "overlaps-line")),
                   "ms": int((time.time() - t_cov) * 1000)}
        others: dict[str, dict[int, list[dict]]] = {e: align_boxes(body, ex) for e, ex in expanded_by_engine.items()}

        # stroke widths -> bold
        widths = [stroke_width(img, ln["bbox"]) if img is not None else None for ln in body]
        split = bold_split([w for w in widths if w])
        bold_flags = [bool(split and w and w > split) for w in widths]
        # a line's column is where its centre falls: a bar read out of the rule
        # can put a right-column line's left edge over the gutter, and the
        # ਮੂਲ|ਅਰਥ row read whole must not open the verse column's stream
        col_of = [next((k for k, (lo, hi) in enumerate(columns) if lo <= (ln["bbox"][0] + ln["bbox"][2]) / 2.0 < hi), 0)
                  if columns else 0 for ln in body]
        streams = (self.stream_matches(body, bold_flags, col_of, hints)
                   if self.index is not None and self.lang == "pa" else {})

        out: list[dict] = []
        for ln in lines:
            rec_out = {"n": ln["n"], "zone": ln["zone"], "bbox": ln["bbox"], "text": normalise(ln.get("text", ""))}
            if ln["zone"] not in ("body", "footnote") or not ln.get("text", "").strip():
                rec_out["dropped"] = ln["zone"] in ("stamp", "pageno", "speck")
                out.append(rec_out)
                continue
            i = body.index(ln)
            cands = [{"engine": self.pivot, "text": ln["text"], "conf": ln.get("conf"),
                      "weight": self.weights.get(self.pivot, 1.0)}]
            raw = {self.pivot: ln["text"]}
            pivot_script = script_of(ln["text"])
            for e, aligned in others.items():
                hits = aligned.get(i, [])
                if not hits:
                    continue
                text = joined_text(hits)
                raw[e] = text
                # a Gurmukhi-only model reading a Latin line emits Gurmukhi
                # noise (measured: 66.6% word accuracy on the Vayakaran, whose
                # introduction is English); it does not vote on such lines
                if pivot_script == "latin" and not LATIN.search(text):
                    continue
                confs = [h["conf"] for h in hits if h.get("conf") is not None]
                cands.append({"engine": e, "text": text, "conf": (sum(confs) / len(confs)) if confs else None,
                              "weight": self.weights.get(e, 1.0)})
            # the pivot read the viewer's stamp as Gurmukhi digits ("। 2੩06 2 0! 530")
            # and the zone pass could not know it; any engine that read it plainly settles it
            if any(is_stamp(t) for t in raw.values()):
                rec_out.update({"zone": "stamp", "dropped": True, "raw": raw})
                out.append(rec_out)
                continue
            w = widths[i]
            bold = bold_flags[i]
            rec_out.update({"col": col_of[i], "bold": bold, "stroke": round(w, 2) if w else None, "raw": raw})
            # how the line came to be: cut from one the engine read across the
            # gutter, or adopted from another engine; the reader of the page
            # (lib/writings_ocr) rejoins split rows, and the report counts both
            if ln.get("split"):
                rec_out["split"] = True
            if ln.get("adopted_from"):
                rec_out["adopted_from"] = ln["adopted_from"]
            if ln.get("recovered"):
                rec_out["recovered"] = ln["recovered"]

            pre = streams.get(i)
            if pre:
                rec_out.update({"kind": "gurbani", "text": pre["text"], "agreement": None,
                                "conf": ln.get("conf"), "protected": True})
                if pre.get("matches"):
                    rec_out["matches"] = pre["matches"]
                if pre.get("merged_into"):
                    rec_out["merged_into"] = pre["merged_into"]
                out.append(rec_out)
                continue

            matches, matched_text = ([], None)
            if self.index is not None and self.lang == "pa":
                quotable = bold or VERSE_MARK in ln["text"]
                matches, matched_text = self.match([c["text"] for c in cands], hints, quotable)
            if matches:
                rec_out.update({"kind": "gurbani", "text": replaced_text(matched_text, matches),
                                "matches": [{k: v for k, v in m.items() if k != "text"} | {"text": m["text"]} for m in matches],
                                "agreement": None, "conf": ln.get("conf"), "protected": True})
                out.append(rec_out)
                continue

            v = vote(cands)
            text = v["text"]
            oov, unknown = (self.lexicon.oov(text) if self.lexicon else (0.0, []))
            corrections: list = []
            if self.corrector is not None and (unknown or self.correct_agreed):
                # only a word the engines disagreed on is corrected: a word every
                # engine read the same way and the lexicon does not know is the
                # author's spelling until proven otherwise
                disputed = disputed_words(ln["text"], [c["text"] for c in cands[1:]])
                # --correct-agreed: an unknown word every engine read alike is
                # offered too, on stricter terms (Corrector.correct_line)
                # (every word, not only lexicon.oov's words of two clusters: ਸੁੀ is one)
                agreed = ({w for w in words(text) if w not in disputed and not self.lexicon.knows(w)}
                          if self.correct_agreed else set())
                if disputed or agreed:
                    text, corrections = self.corrector.correct_line(text, only=disputed, agreed=agreed or None)
                    if corrections:
                        oov, unknown = self.lexicon.oov(text)
            has_verse = VERSE_MARK in text
            if bold and _RAAG_HEADING.search(text) and len(words(text)) <= HEADING_MAX_WORDS:
                kind = "heading"                            # "ਸਿਰੀ ਰਾਗੁ ਮਹਲਾ ੧ ॥": bibliographic, kept out of the index
            elif bold and has_verse:
                kind = "gurbani-unmatched"
            elif bold and len(words(text)) <= HEADING_MAX_WORDS:
                kind = "heading"
            else:
                kind = "commentary"
            rec_out.update({"kind": kind, "text": text, "agreement": v["agreement"], "conf": v["conf"],
                            "engines": v["n"], "oov": oov, "oov_words": unknown[:12], "n_words": len(words(text)),
                            "corrections": corrections})
            rec_out["route"] = needs_arbiter(rec_out, self.lang)
            if ln.get("recovered") and self.coverage and self.coverage.get("route"):
                # a crop read on its own is routed for a second look whatever
                # its numbers say, when the measurement asked for that
                rec_out["route"] = rec_out["route"] or "recovered"
            out.append(rec_out)

        page_route = page_needs_arbiter([r for r in out if r.get("zone") == "body"], self.lang)
        meta = {"page": page, "hints": hints, "hints_raw": raw_hints,
                "columns": columns, "columns_source": columns_source, "rule": vrule,
                "engines": [self.pivot] + list(others),
                "pivot": self.pivot, "body_h": typical_h, "bold_split": round(split, 2) if split else None,
                "page_w": pmeta["page_w"], "page_h": pmeta["page_h"], "routed": page_route, "coverage": cov}
        return meta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--engines", help="comma-separated; default every engine with output")
    ap.add_argument("--pivot")
    ap.add_argument("--weights", default="auto",
                    help="engine=weight,... or 'auto': each engine's measured word accuracy to the fourth "
                         "power (data/raw/ocr-eval-<book>.json), 0.5 where unmeasured")
    ap.add_argument("--pages")
    ap.add_argument("--no-correct", action="store_true")
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--corpus", default=CORPUS_DB)
    ap.add_argument("--granths", default=GRANTHS_DB)
    ap.add_argument("--scriptures", default=SCRIPTURES_FULL_DB,
                    help="the full scripture store (data/scriptures-full.sqlite); preferred to --granths where it exists")
    ap.add_argument("--kosh", default=MAHANKOSH_DB)
    ap.add_argument("--out", default=OCR_DIR)
    ap.add_argument("--out-name", default="merged",
                    help="write to data/ocr/<book>/<name>/ instead of merged/, so two merges of one book can be "
                         "scored side by side (24_ocr_eval.py --merged); must start with 'merged'")
    ap.add_argument("--coverage", dest="coverage", action="store_true", default=False,
                    help="find the ink no line covers and read it from crops (lib/ocr_coverage); "
                         "every region is recorded in _meta.coverage")
    ap.add_argument("--no-coverage", dest="coverage", action="store_false")
    ap.add_argument("--coverage-psm", type=int, default=7, help="Tesseract page segmentation mode for a crop (7: one line)")
    ap.add_argument("--coverage-max", type=int, default=MAX_REGIONS,
                    help="crops per page at most (default %d); the shape tests run first, so a picture page costs little" % MAX_REGIONS)
    ap.add_argument("--coverage-min-agreement", type=float, default=0.0,
                    help="refuse a recovered line the other Tesseract variants agree with below this (0: off)")
    ap.add_argument("--coverage-route", action="store_true", help="route every recovered line for a second look")
    ap.add_argument("--correct-agreed", action="store_true",
                    help="also correct an unknown word every engine read alike, when it is one seeded confusion from "
                         "a prose word (a grey scan's shared misreadings of the subjoined ra); default: disputed words only")
    args = ap.parse_args()
    if not args.out_name.startswith("merged") or "/" in args.out_name:
        sys.exit("--out-name must be 'merged' or 'merged-<something>': the evaluation finds merges by that prefix")

    book_dir = os.path.join(args.out, args.book)
    with open(os.path.join(book_dir, "pages.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    lang = meta.get("language", "en")
    engines = [e.strip() for e in args.engines.split(",")] if args.engines else default_engines(book_dir)
    if not engines:
        sys.exit("no engine output under %s/ocr" % book_dir)
    pivot = args.pivot or default_pivot(engines, lang)
    if pivot not in engines:
        engines.insert(0, pivot)
    acc = measured_accuracy(args.book) if args.weights == "auto" else {}
    if args.weights == "auto" and not args.engines:
        engines, dropped = choose_voters(engines, acc, pivot)
        if dropped:
            print("not voting (measured below %.0f%% word accuracy): %s; name them with --engines to force it"
                  % (MIN_VOTER_ACC * 100, ", ".join("%s %.1f%%" % (e, acc[e] * 100) for e in dropped)))
    weights = auto_weights(acc, engines) if args.weights == "auto" else {}
    for part in args.weights.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            weights[k.strip()] = float(v)

    t0 = time.time()
    index = load_index(args.corpus, args.granths, args.scriptures) if lang == "pa" else None
    lexicon = None if args.no_lexicon else Lexicon.from_sources(args.kosh if lang == "pa" else None, args.corpus)
    corrector = None if (args.no_correct or lexicon is None) else Corrector(lexicon, Confusions.load(CONFUSIONS))
    print("%s (%s): engines %s, pivot %s; corpus %s lines; lexicon %s; %.0fs to load"
          % (args.book, lang, ",".join(engines), pivot, len(index) if index else 0,
             lexicon.sources if lexicon else "off", time.time() - t0))

    coverage = ({"psm": args.coverage_psm, "max": args.coverage_max, "min_agreement": args.coverage_min_agreement,
                 "route": args.coverage_route} if args.coverage else None)
    merger = Merger(book_dir, meta, engines, pivot, weights, index, lexicon, corrector, coverage,
                    correct_agreed=args.correct_agreed)
    # every page's header, whatever --pages asks for: a page's ang is judged by its neighbours
    merger.header_pass(meta["pages"])
    fixed = sum(1 for p, h in merger.hints.items() if h.get("ang_source") == "smoothed")
    print("headers: %d pages read, %d angs put right or filled in by their neighbours"
          % (sum(1 for h in merger.hints_raw.values() if h.get("ang_from")), fixed))
    if lexicon is not None:
        print("book vocabulary: %d words two engines agree on" % merger.vocab_pass(meta["pages"]))
    wanted = set(n + 1 for n in parse_pages(args.pages, max(p["page"] for p in meta["pages"])))
    merged_dir = os.path.join(book_dir, args.out_name)
    os.makedirs(merged_dir, exist_ok=True)
    stats: Counter = Counter()
    agreements, oovs, routed_pages = [], [], []
    cov_ms: list[int] = []
    uncovered_pages: list[int] = []
    done = 0
    t0 = time.time()
    for rec in meta["pages"]:
        if rec["page"] not in wanted:
            continue
        got = merger.merge_page(rec)
        if got is None:
            continue
        pmeta, lines = got
        write_page(os.path.join(merged_dir, "%04d.jsonl" % rec["page"]), pmeta, lines)
        done += 1
        stats["columns:" + (pmeta.get("columns_source") or "none")] += 1
        stats["ang:" + ((pmeta.get("hints") or {}).get("ang_source") or "none")] += 1
        cov = pmeta.get("coverage") or {}
        if cov.get("on"):
            stats["coverage:pages-checked"] += 1
            if cov["regions"]:
                stats["coverage:pages-with-residual"] += 1
            stats["coverage:regions"] += len(cov["regions"])
            stats["coverage:recovered"] += cov["recovered"]
            stats["coverage:uncovered-after"] += cov["uncovered_after"]
            stats["coverage:capped-pages"] += int(cov["capped"])
            for why, n in cov["rejected"].items():
                stats["coverage:rejected:" + why] += n
            cov_ms.append(cov["ms"])
            if cov["uncovered_after"]:
                uncovered_pages.append(rec["page"])
        for ln in lines:
            stats["zone:" + ln["zone"]] += 1
            if ln.get("split"):
                stats["split-at-gutter"] += 1
            if ln.get("adopted_from"):
                stats["adopted"] += 1
            if ln.get("recovered"):
                stats["recovered"] += 1
            if ln.get("kind"):
                stats["kind:" + ln["kind"]] += 1
            if ln.get("agreement") is not None:
                agreements.append(ln["agreement"])
            if ln.get("oov") is not None:
                oovs.append(ln["oov"])
            stats["corrections"] += len(ln.get("corrections") or [])
            stats["corrections-agreed"] += sum(1 for c in ln.get("corrections") or [] if c[2] == "confusion-agreed")
            if ln.get("route"):
                stats["routed-lines"] += 1
                stats["route:" + ln["route"]] += 1
        if pmeta["routed"]:
            routed_pages.append({"page": rec["page"], **pmeta["routed"]})
        if done % 20 == 0:
            print("  %d pages (%.1fs/page)" % (done, (time.time() - t0) / done))

    route = {"pages": routed_pages, "engines": engines, "pivot": pivot}
    with open(os.path.join(merged_dir, "route.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(route, fh, ensure_ascii=False, indent=1)
    report = {"book": args.book, "language": lang, "engines": engines, "pivot": pivot, "weights": weights, "pages": done,
              "counts": dict(stats),
              "agreement_mean": round(statistics.mean(agreements), 3) if agreements else None,
              "agreement_p10": round(sorted(agreements)[len(agreements) // 10], 3) if agreements else None,
              "oov_mean": round(statistics.mean(oovs), 3) if oovs else None,
              "routed_pages": len(routed_pages), "correct": corrector is not None,
              "correct_agreed": bool(args.correct_agreed),
              "lexicon": lexicon.sources if lexicon else None, "seconds": round(time.time() - t0, 1),
              "coverage": ({**coverage, "pages_with_uncovered_after": uncovered_pages[:100],
                            "ms_per_page": round(statistics.mean(cov_ms), 1) if cov_ms else None}
                           if coverage else {"on": False})}
    os.makedirs(RAW, exist_ok=True)
    # a named merge keeps its own report, so a comparison never overwrites the
    # numbers of the merge that is actually ingested
    suffix = "" if args.out_name == "merged" else "-" + args.out_name[len("merged-"):]
    with open(os.path.join(RAW, "ocr-report-%s%s.json" % (args.book, suffix)), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("lexicon",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
