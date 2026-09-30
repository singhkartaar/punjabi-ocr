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
data/raw/ocr-report-<book>.json. Nothing here is sent anywhere.
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
from lib.ocr_correct import Confusions, Corrector
from lib.ocr_engines import read_page, write_page
from lib.ocr_layout import bold_split, stroke_width
from lib.ocr_lexicon import Lexicon
from lib.ocr_match import CorpusIndex, align_stream, match_text_all
from lib.ocr_pages import parse_pages
from lib.ocr_route import needs_arbiter, page_needs_arbiter
from lib.ocr_text import GURMUKHI, LATIN, key_positions, normalise, script_of, splice, words
from lib.ocr_vote import align_boxes, is_block, joined_text, segment_block, vote
from lib.ocr_zones import classify_zones, footnote_rule_y, header_of, is_stamp, page_columns
from lib.paths import CORPUS_DB, GRANTHS_DB, MAHANKOSH_DB, OCR_DIR, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CONFUSIONS = os.path.join(OCR_DIR, "confusions.json")
RAW = os.path.join(ROOT, "data", "raw")
VERSE_MARK = "॥"
HEADING_MAX_WORDS = 6
# a plain prose line must be mostly verse for a contained match to count as a quotation
PROSE_COVERAGE = 0.6
_RAAG_HEADING = re.compile(r"ਮਹਲਾ\s*[੧-੯ਮ]|ਮਃ\s*[੧-੯]")   # ਮਹਲਾ ੧ / ਮਃ ੧


def load_index(corpus: str, granths: str) -> CorpusIndex | None:
    rows = []
    if corpus and os.path.exists(corpus) and os.path.getsize(corpus) > 0:
        con = sqlite3.connect(corpus)
        # headings ("ਮਾਰੂ ਮਹਲਾ ੫ ॥") are bibliographic and would match the
        # citations the books print after a quote
        rows += [(*r, "G") for r in con.execute(
            "SELECT line_id, shabad_id, ang, gurmukhi_uni FROM lines WHERE kind != 'heading' ORDER BY line_id")]
        con.close()
    if granths and os.path.exists(granths) and os.path.getsize(granths) > 0:
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
    """
    keep, drop = [], []
    for e in engines:
        if e != pivot and e in acc and acc[e] < MIN_VOTER_ACC:
            drop.append(e)
        else:
            keep.append(e)
    return keep, drop


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
    """
    adopted: list[dict] = []
    for engine, lines in others.items():
        for ln in lines:
            if not ln.get("text", "").strip() or (ln["bbox"][3] - ln["bbox"][1]) > 2.5 * typical_h:
                continue
            covered = any(_overlap_v(ln["bbox"], p["bbox"]) >= 0.4 for p in pivot_body + adopted)
            if not covered:
                adopted.append({**ln, "words": ln.get("words") or [], "adopted_from": engine})
    return adopted


def _overlap_v(a: list, b: list) -> float:
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iy / max(1, min(a[3] - a[1], b[3] - b[1]))


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


def split_at_gutter(lines: list[dict], at: int, tol: int = 20) -> list[dict]:
    """
    A body line the engine read across both columns, cut at the gutter by its
    word boxes. Tesseract's page segmentation usually keeps the columns apart;
    where a hairline rule fooled it, "ਖਲਾਵੈ ਕਵਣੁ ... | ਚੁਗਾਉਂਦਾ ਹੈ?" is two lines.
    """
    out = []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        ws = ln.get("words") or []
        if ln.get("zone") not in ("body", "footnote") or not ws or not (x0 < at - tol and x1 > at + tol):
            out.append(ln)
            continue
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


class Merger:
    def __init__(self, book_dir: str, meta: dict, engines: list[str], pivot: str, weights: dict,
                 index: CorpusIndex | None, lexicon: Lexicon | None, corrector: Corrector | None):
        self.book_dir, self.meta = book_dir, meta
        self.engines, self.pivot, self.weights = engines, pivot, weights
        self.index, self.lexicon, self.corrector = index, lexicon, corrector
        self.lang = meta.get("language", "en")
        self.sources = ["G"] + ([meta["scripture"]] if meta.get("scripture", "G") != "G" else [])
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
        hints = header_of(lines)
        wordboxes = [w for ln in lines for w in ln.get("words", [])] or lines
        columns = [] if self.notation else page_columns(wordboxes, pmeta["page_w"], img, lines)
        if columns:
            lines = split_at_gutter(lines, columns[0][1])
        body = [ln for ln in lines if ln["zone"] in ("body", "footnote") and ln.get("text", "").strip()]
        heights = sorted(ln["bbox"][3] - ln["bbox"][1] for ln in body) or [40]
        typical_h = heights[len(heights) // 2]

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
        others: dict[str, dict[int, list[dict]]] = {e: align_boxes(body, ex) for e, ex in expanded_by_engine.items()}

        # stroke widths -> bold
        widths = [stroke_width(img, ln["bbox"]) if img is not None else None for ln in body]
        split = bold_split([w for w in widths if w])
        bold_flags = [bool(split and w and w > split) for w in widths]
        col_of = [next((k for k, (lo, hi) in enumerate(columns) if lo <= ln["bbox"][0] < hi), 0) if columns else 0
                  for ln in body]
        streams = (self.stream_matches(body, bold_flags, col_of, hints)
                   if self.index is not None and self.lang == "pa" else {})

        out: list[dict] = []
        for ln in lines:
            rec_out = {"n": ln["n"], "zone": ln["zone"], "bbox": ln["bbox"], "text": normalise(ln.get("text", ""))}
            if ln["zone"] not in ("body", "footnote") or not ln.get("text", "").strip():
                rec_out["dropped"] = ln["zone"] in ("stamp", "pageno")
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
            if self.corrector is not None and unknown:
                # only a word the engines disagreed on is corrected: a word every
                # engine read the same way and the lexicon does not know is the
                # author's spelling until proven otherwise
                disputed = disputed_words(ln["text"], [c["text"] for c in cands[1:]])
                if disputed:
                    text, corrections = self.corrector.correct_line(text, only=disputed)
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
            out.append(rec_out)

        page_route = page_needs_arbiter([r for r in out if r.get("zone") == "body"], self.lang)
        meta = {"page": page, "hints": hints, "columns": columns, "engines": [self.pivot] + list(others),
                "pivot": self.pivot, "body_h": typical_h, "bold_split": round(split, 2) if split else None,
                "page_w": pmeta["page_w"], "page_h": pmeta["page_h"], "routed": page_route}
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
    ap.add_argument("--kosh", default=MAHANKOSH_DB)
    ap.add_argument("--out", default=OCR_DIR)
    args = ap.parse_args()

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
    index = load_index(args.corpus, args.granths) if lang == "pa" else None
    lexicon = None if args.no_lexicon else Lexicon.from_sources(args.kosh if lang == "pa" else None, args.corpus)
    corrector = None if (args.no_correct or lexicon is None) else Corrector(lexicon, Confusions.load(CONFUSIONS))
    print("%s (%s): engines %s, pivot %s; corpus %s lines; lexicon %s; %.0fs to load"
          % (args.book, lang, ",".join(engines), pivot, len(index) if index else 0,
             lexicon.sources if lexicon else "off", time.time() - t0))

    merger = Merger(book_dir, meta, engines, pivot, weights, index, lexicon, corrector)
    wanted = set(n + 1 for n in parse_pages(args.pages, max(p["page"] for p in meta["pages"])))
    merged_dir = os.path.join(book_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    stats: Counter = Counter()
    agreements, oovs, routed_pages = [], [], []
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
        for ln in lines:
            stats["zone:" + ln["zone"]] += 1
            if ln.get("kind"):
                stats["kind:" + ln["kind"]] += 1
            if ln.get("agreement") is not None:
                agreements.append(ln["agreement"])
            if ln.get("oov") is not None:
                oovs.append(ln["oov"])
            stats["corrections"] += len(ln.get("corrections") or [])
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
              "lexicon": lexicon.sources if lexicon else None, "seconds": round(time.time() - t0, 1)}
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "ocr-report-%s.json" % args.book), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("lexicon",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
