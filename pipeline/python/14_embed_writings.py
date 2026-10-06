"""
Build the writings corpus: retrieval units, their vectors, their citations.

  14_embed_writings.py
  14_embed_writings.py --corpus writings-en --target-tokens 60
  14_embed_writings.py --src data/treatises --corpus treatises-pa \
      --model multilingual-e5-small-gurbani-trim --lang pa --max-len 160
  14_embed_writings.py --lang pa                       # the Punjabi works of data/writings
  14_embed_writings.py --lang en --translations        # the English ones, plus every translated one
  14_embed_writings.py --src data/barusahib --corpus barusahib-en --lang en --translations \
      --keep artifacts/barusahib.sqlite                # a scanned book joins a corpus built elsewhere

Writes <src>/units.jsonl (what a reader is shown and what the server serves;
units-<lang>.jsonl for a language other than English, so the Punjabi and the
English of one folder never write the same file) and artifacts/corpora/<corpus>/
(what retrieval reads).

One script, several corpora. What differs between an author is the source
directory, the model that can read the script the prose is written in, and the
quote policy carried per work in the source's own `_meta`; nothing below is
specific to one of them. The example above is Prof. Sahib Singh's Punjabi
prose, whose model is the one the Gurmukhi indexes already load, so a second
corpus costs no additional model memory on the VM. A folder of scanned books
may hold works in more than one language (lib/writings_manifest.py); a run
takes the works of `--lang`, and `--translations` also takes the works of
another language whose English 26_translate_writings.py has written.

Two decisions are made here and both are load-bearing.

**A source paragraph is not a retrieval unit**, and it is not one in either
direction. Nearly half of Bau Ji's paragraphs are ten tokens or fewer -- he
writes in cascading indented lists ("studied or cause to be studied / learnt or
taught / taken or given") -- and a unit reading "recognise" retrieves nothing
and means nothing, so paragraphs are merged forward to a target size. Sahib
Singh writes 200-word paragraphs, which overflow the model's window and would
be averaged into one blurred vector, so a paragraph past the cap is cut at its
own sentence ends instead. Neither merge nor cut crosses a work, a part or a
heading. The merge also repairs the 4.4% of paragraphs the PDF reader splits
mid-sentence, which is why it over-segments there rather than under.

**The quoted Gurbani is not part of the unit's text.** A quotation is Gurbani's
words, already searchable through the Gurbani indexes over the whole Granth;
folding it in would make a corpus retrieve for verses instead of for what its
author says about them, and the two would stop doing different jobs. The
quotation becomes a CITATION on the unit that leads into it, carrying the shabad
id, so an answer can quote the exposition and show the verse beneath it.

The corpus lives under artifacts/corpora/, which is deliberately one level
deeper than artifacts/<index>/: registry.js, test_pipeline.py, the integration
test, 08_annotate_manifest.py and .dockerignore all scan exactly one level, so
a corpus is invisible to every one of them and cannot be mistaken for a Gurbani
index whose ids address the Granth. Its manifest says so as well, `"kind":
"documents"`, and the registry refuses one wherever it finds it. The server
pairs the directory with the database 15_build_writings_db.py writes through
its CORPORA table (key, dir, db), and the public search API serves the pair
as a `writings-<key>` data pack.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.documents import embed_long
from lib.embedder import Embedder, BATCH_SIZE, build_env
from lib.models import first_available, profile
from lib.paths import ARTIFACTS, ROOT
from lib.quantize import quantize
from lib.writings_works import db_name

WRITINGS = os.path.join(ROOT, "data", "writings")
OUT_DIM = 256


def index_dim_for(n_units: int, embed_dim: int, out_dim: int = OUT_DIM) -> int:
    """
    The dimension the PCA reduces to: OUT_DIM, or fewer when the corpus has
    fewer units than that. A work of a few pages -- a word-meaning list, a
    bench sample of twenty pages -- gives 77 units, and a 256-component PCA
    over 77 rows is not a thing; the manifest records the dimension used and
    the reader takes it from there.
    """
    return max(1, min(out_dim, n_units, embed_dim))
# What closes a sentence, so that a merge never leaves one hanging. Punjabi
# prose ends on a danda, not a full stop: without these two characters every
# unit of a Gurmukhi work runs to the hard cap instead of the target, and 4,703
# paragraphs collapse into 895 passages of 213 words each.
ENDS_SENTENCE = tuple(".!?:;–—”’\")" + "।॥")
# Where a paragraph may be cut into units: after a sentence mark, never inside
# one. The same expression lib/documents.py chunks on.
SENTENCE = re.compile(r"(?<=[।॥.?!])\s+")
# One corpus per language: the English essays embed with the English model,
# a Punjabi book with the Gurbani-tuned multilingual one, and each corpus
# keeps its own units file and unit_row space. Gurmukhi prose is embedded with
# a model that reads both scripts, so a reader may ask it in either; English
# prose is bge and Latin only.
LABELS = {"en": "Writings · English", "pa": "Writings · Punjabi", "hi": "Writings · Hindi"}
LABELS_PA = {"en": "ਲਿਖਤਾਂ · ਅੰਗਰੇਜ਼ੀ", "pa": "ਲਿਖਤਾਂ · ਪੰਜਾਬੀ", "hi": "ਲਿਖਤਾਂ · ਹਿੰਦੀ"}
LANG_DEFAULTS = {
    "en": {"corpus": "writings-en", "models": ["bge-small-en-v1.5"], "query_scripts": ["latin"]},
    # the Gurbani fine-tune when it is on this machine, else the base model it
    # was tuned from (same tokenizer, same settings; a public checkout has only the base)
    "pa": {"corpus": "writings-pa", "models": ["multilingual-e5-small-gurbani-trim", "multilingual-e5-small"],
           "query_scripts": ["gurmukhi", "latin"]},
    "hi": {"corpus": "writings-hi", "models": ["multilingual-e5-small-gurbani-trim", "multilingual-e5-small"],
           "query_scripts": ["devanagari", "latin"]},
}


def sentences(text: str) -> list[str]:
    """
    A paragraph cut after each sentence mark, keeping the mark.

    The same rule lib/documents.py splits on, so a piece that survives here
    fits the same way there: a full stop, a question or exclamation mark, or
    either danda, followed by space.
    """
    parts = SENTENCE.split(text)
    return [p for p in (s.strip() for s in parts) if p] or [text]


def units_name(lang: str) -> str:
    """The units file of a language: units.jsonl for English, units-<lang>.jsonl otherwise."""
    return "units.jsonl" if lang == "en" else "units-%s.jsonl" % lang


def load_work(path: str):
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return rows[0]["_meta"], [r for r in rows[1:]]


def apply_translations(records: list[dict], path: str) -> tuple[list[dict], int, int]:
    """
    The work's prose in English, its Punjabi kept beside it.

    26_translate_writings.py writes <work>.en.jsonl, one English record per
    prose paragraph. Here every body/heading/footnote record takes that English
    as its `text` and keeps the original as `text_src`; a paragraph with no
    translation (rejected, or not yet run) is dropped rather than embedded in
    Gurmukhi into an English corpus. Quote records pass through untouched --
    they are Gurbani, and their citations resolve the same either way.

    @returns (records, translated, dropped)
    """
    en: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "_meta" not in rec and rec.get("en"):
                en[rec["unit_id"]] = rec["en"]
    out, translated, dropped = [], 0, 0
    for rec in records:
        if rec.get("style") == "quote" or rec.get("lang") == "en":
            out.append(rec)
            continue
        text = en.get(rec["unit_id"])
        if not text:
            dropped += 1
            continue
        out.append({**rec, "text": text, "text_src": rec["text"], "lang": "en", "lang_src": rec.get("lang", "pa")})
        translated += 1
    return out, translated, dropped


def kept_corpus(db_path: str, rebuilt: set[str]) -> tuple[list[dict], list[dict]]:
    """
    The works of a built corpus that are not being rebuilt, taken as they were
    built: their units, text and citations from its database, in its order.

    For a corpus whose sources are on another machine. Baru Sahib's roster
    reads PDFs that live on the machine that published the corpus; a scanned
    book joining it (manifest `corpus`) is OCR'd here. With --keep, every work
    whose <work>.jsonl is in --src is read from it as always, and only the
    others come from the database -- so where every source is present, as on
    the publishing machine, --keep changes nothing. The kept units are
    embedded again from their own text (the same model gives the same vector)
    so that one PCA is fitted over the whole corpus.

    A database from before `links` existed carries its Guru Granth Sahib
    citations only; those are what a kept unit's cites are rebuilt from.
    """
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    con.row_factory = sqlite3.Row
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    work_cols = {r[1] for r in con.execute("PRAGMA table_info(works)")}
    unit_cols = {r[1] for r in con.execute("PRAGMA table_info(units)")}
    # a database from before these columns existed (the English corpora built
    # in September) simply has none: the work is as it was, with nothing said
    opt = lambda row, cols, k: row[k] if k in cols else None
    works = []
    for w in con.execute("SELECT * FROM works ORDER BY rowid"):
        if w["work_id"] in rebuilt:
            continue
        works.append({"work": w["work_id"], "title": w["title"], "title_en": w["title_en"], "author": w["author"],
                      "folder": opt(w, work_cols, "folder"), "original": bool(w["original"]), "quote_policy": w["quote_policy"],
                      "parts": json.loads(w["parts"] or "[]"), "files": opt(w, work_cols, "files"),
                      "language": opt(w, work_cols, "language"), "licence": opt(w, work_cols, "licence"), "kept": True,
                      **{k: w[k] for k in ("kind", "translate", "ang_from", "ang_to") if k in work_cols}})
    meta = {w["work"]: w for w in works}
    cites: dict[int, list[dict]] = {}
    if "links" in tables:
        for c in con.execute("SELECT * FROM links ORDER BY rowid"):
            cites.setdefault(c["unit_row"], []).append(
                {"source": c["source"], "shabad_id": c["shabad_id"], "line_id": c["line_from"],
                 "line_from": c["line_from"], "line_to": c["line_to"], "ang": c["ang"], "role": c["role"],
                 "method": c["method"], "score": c["score"], "page": c["page"], "span": ""})
        # the span is the citations table's alone
        spans = {(c["unit_row"], c["line_id"]): c["span"] for c in con.execute("SELECT * FROM citations")}
        for row, cs in cites.items():
            for c in cs:
                c["span"] = spans.get((row, c["line_id"]), "")
    else:
        for c in con.execute("SELECT * FROM citations ORDER BY rowid"):
            cites.setdefault(c["unit_row"], []).append(
                {"source": "G", "shabad_id": c["shabad_id"], "line_id": c["line_id"], "line_from": c["line_id"],
                 "line_to": c["line_id"], "ang": c["ang"], "role": "quotes", "method": c["method"],
                 "score": c["score"], "page": None, "span": c["span"]})
    units = []
    for u in con.execute("SELECT * FROM units ORDER BY unit_row"):
        w = meta.get(u["work_id"])
        if w is None:
            continue
        units.append({"work": u["work_id"], "part": u["part"], "page": u["page"], "para_no": u["para_no"],
                      "marker": u["marker"], **({"section": u["section"]} if opt(u, unit_cols, "section") else {}),
                      "text": u["text"], "text_src": opt(u, unit_cols, "text_src"), "cites": cites.get(u["unit_row"], []),
                      "unit_id": u["unit_id"], "title": w["title"], "title_en": w["title_en"],
                      "author": w["author"], "quote_policy": w["quote_policy"]})
    con.close()
    for w in works:
        mine = [u for u in units if u["work"] == w["work"]]
        w["units"] = len(mine)
        w["cites"] = w["links"] = sum(len(u["cites"]) for u in mine)
    return works, units


def build_units(records: list[dict], citations: dict, target: int, hard: int) -> list[dict]:
    """
    Paragraphs merged into retrieval units, with the verses they lead into.

    A unit closes once it is big enough AND ends on a full stop, so a merge
    never leaves a sentence hanging; it closes regardless at `hard` tokens, at a
    heading, and at a part boundary.

    A verse's links go BACK to the prose that led into it (an essay quotes a
    tuk to make its point), unless some prose EXPLAINS the verse's pair -- a
    translation, a word-meaning, a commentary -- and then they go FORWARD to
    that prose, in the "explains" role, with the links the prose itself
    states. Each cite carries `role`, `source` and its line range.
    """
    # One paragraph may cite SEVERAL verses, so the map holds a list per
    # paragraph. A bare citation is accepted and wrapped rather than trusted to
    # be a list: iterating a dict yields its KEYS, so a caller passing the older
    # one-per-paragraph shape would have quietly indexed strings as citations.
    citations = {k: (v if isinstance(v, list) else [v]) for k, v in (citations or {}).items()}
    units: list[dict] = []
    cur: dict | None = None
    carried: list[dict] = []      # citations from a unit that was discarded
    # The pairs some prose EXPLAINS (lib/writings_ocr: the arth beside a verse,
    # or after it in a translation). A verse of such a pair is not the prose
    # before it leading into a quotation; it is what the prose after it is
    # about, so its citations wait for that prose and attach there, forward.
    explained = {e["pair"] for rec in records for e in (rec.get("explains") or []) if e.get("pair")}
    held: dict[int, list[dict]] = {}          # pair -> the verse's citations, until its prose comes

    def add_cite(unit, cite):
        # One shabad, once, per scripture and role: a source may name the same
        # shabad after every paragraph of a gist -- which is how a long gist
        # stays linked when the embedder splits it -- and a passage listing it
        # twice is noise. Two mentions of one shabad widen the range to cover
        # both: an arth that explains tuks 3-4 and then 5-6 explains 3-6.
        key = (cite.get("source") or "G", cite["shabad_id"], cite.get("role") or "quotes")
        for c in unit["cites"]:
            if (c.get("source") or "G", c["shabad_id"], c.get("role") or "quotes") == key:
                if cite.get("line_from") is not None and c.get("line_from") is not None:
                    c["line_from"] = min(c["line_from"], cite["line_from"])
                    c["line_to"] = max(c["line_to"], cite["line_to"])
                return
        unit["cites"].append(dict(cite))

    def explains_cites(rec):
        """The links a prose paragraph's `explains` state, as citations in the explains role."""
        out = []
        for e in rec.get("explains") or []:
            for cite in held.pop(e.get("pair"), []):
                out.append({**cite, "role": "explains"})
            if e.get("shabad_id") is None:
                continue                       # verse the merge could not place: the pair links nothing yet
            out.append({"shabad_id": e["shabad_id"], "line_id": e.get("line_from"),
                        "line_from": e.get("line_from"), "line_to": e.get("line_to"),
                        "source": e.get("source") or "G", "role": "explains", "page": rec.get("page"),
                        "ang": e.get("ang"), "score": e.get("score"), "method": e.get("method") or "ocr-corpus-match",
                        "span": ""})
        return out

    def flush():
        # A unit needs PROSE. A heading carried into the next unit is context;
        # a heading standing alone is a retrieval unit that says only what a
        # section is called, and it wins queries about its own subject while
        # telling the reader nothing -- "ਸਲੋਕ ਭਗਤ ਕਬੀਰ ਜੀਉ ਕੇ" came back as a
        # passage about ego before this counted body words separately.
        #
        # A discarded unit must not take its verses with it: a heading followed
        # directly by a quotation is exactly how these works open a section, and
        # dropping the unit silently dropped 205 citations of the English corpus
        # before this carried them forward.
        nonlocal cur
        if cur and cur["body"] >= 3:
            cur["text"] = " ".join(cur["text"]).strip()
            # an untranslated work carries no source text at all
            if cur["text_src"]:
                cur["text_src"] = " ".join(cur["text_src"]).strip()
            else:
                del cur["text_src"]
            del cur["words"], cur["body"]
            units.append(cur)
        elif cur and cur["cites"]:
            # forward, not back: a heading with a verse under it is a section
            # opening, and what explains that verse is the prose that follows
            carried.extend(cur["cites"])
        cur = None

    def start(rec):
        held, carried[:] = list(carried), []
        return {"work": rec["work"], "part": rec["part"], "page": rec["page"],
                "para_no": rec["para_no"], "marker": rec.get("marker"),
                **({"section": rec["section"]} if rec.get("section") else {}),
                "text": [], "text_src": [], "words": 0, "body": 0, "cites": held}

    for rec in records:
        found = citations.get(rec["unit_id"]) or []
        if rec["style"] == "quote":
            if rec.get("pair") in explained:
                # the verse is what the prose after it explains: its links wait there
                held.setdefault(rec["pair"], []).extend(found)
                continue
            # the verse belongs to the prose that introduced it
            for cite in found:
                if cur:
                    add_cite(cur, cite)
                elif units:
                    add_cite(units[-1], cite)
            continue
        # A verse can also sit INSIDE a prose paragraph rather than standing as
        # its own -- Bhai Raghbir Singh quotes a tuk mid-sentence, and the
        # resolver deliberately leaves such a paragraph as prose so his own
        # words are still retrievable. The citation still belongs to it, and
        # before this only the paragraphs restyled as quotes were linked at all:
        # 35 of 80 for Bandginama.
        pending = found + explains_cites(rec)
        # The source-language text of a translated paragraph (apply_translations)
        # goes whole to the unit its first sentence lands in: the English is
        # cut at its own sentence ends below, and the Punjabi does not break
        # where its translation does.
        src_pending = rec.get("text_src")
        # a heading closes the unit before it, but only once that unit has prose
        # of its own: consecutive headings belong to the passage they introduce
        if rec["style"] == "heading" and cur is not None and cur["body"] > 0:
            flush()
        if cur is not None and (cur["work"], cur["part"]) != (rec["work"], rec["part"]):
            flush()
        # A paragraph longer than the cap is cut at its own sentence ends and
        # each piece becomes a unit in its own right. It is NOT handed whole to
        # the embedder: over the model's window embed_long averages the chunks,
        # and an averaged passage is the blurred centroid lib/documents.py says
        # it is. Bau Ji's paragraphs are short enough that this never fires;
        # Sahib Singh writes 200-word paragraphs and it fires often.
        for piece in sentences(rec["text"]):
            if cur is None:
                cur = start(rec)
            if pending:
                for cite in pending:
                    add_cite(cur, cite)
                pending = []
            if src_pending:
                cur["text_src"].append(src_pending)
                src_pending = None
            cur["text"].append(piece)
            cur["words"] += len(piece.split())
            if rec["style"] != "heading":
                cur["body"] += len(piece.split())
            whole = piece.rstrip().endswith(ENDS_SENTENCE)
            if cur["words"] >= hard or (cur["words"] >= target and whole):
                flush()
    flush()
    # a verse quoted after the last prose of the work has nothing to carry to;
    # a verse whose explanation never came (it was said to have one) likewise
    for cites in held.values():
        carried.extend(cites)
    if carried and units:
        for cite in carried:
            add_cite(units[-1], cite)
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=WRITINGS,
                    help="directory of <work>.jsonl + citations.jsonl (default data/writings)")
    ap.add_argument("--lang", "--text-lang", dest="lang", default="en", choices=sorted(LANG_DEFAULTS),
                    help="the language of the prose: which works to embed (their _meta.language; a work "
                         "that states none is taken to be this), and the corpus, model and query-script defaults")
    ap.add_argument("--corpus", help="default: writings-<lang>")
    ap.add_argument("--model", help="default: the language's model (LANG_DEFAULTS)")
    ap.add_argument("--units-out", help="default: <src>/units.jsonl for en, units-<lang>.jsonl otherwise")
    ap.add_argument("--out-dir", help="default: artifacts/corpora/<corpus>")
    ap.add_argument("--max-len", type=int, default=256, help="token window for a unit")
    ap.add_argument("--target-tokens", type=int, default=60, help="words a unit aims for")
    ap.add_argument("--hard-tokens", type=int, default=220, help="words a unit never exceeds")
    ap.add_argument("--keep", metavar="DB",
                    help="a built corpus database: its works with no <work>.jsonl in --src are carried over "
                         "as built (kept_corpus), the rest read from --src")
    ap.add_argument("--units-only", action="store_true",
                    help="write units.jsonl and stop: for re-linking a corpus whose text is unchanged "
                         "(35_link_scriptures.py), where the published vectors still hold -- compare the "
                         "units with the built database before 15_build_writings_db.py rebuilds it")
    ap.add_argument("--translations", action="store_true",
                    help="also take every work of another language that has a <work>.en.jsonl "
                         "(26_translate_writings.py), embedding its English into this corpus; --lang en")
    args = ap.parse_args()
    if args.translations and args.lang != "en":
        sys.exit("--translations embeds English: use it with --lang en")
    defaults = LANG_DEFAULTS[args.lang]
    args.corpus = args.corpus or defaults["corpus"]
    if not args.model:
        args.model = first_available(defaults["models"])
        if not args.model:
            sys.exit("no model for %s under vendor/models: one of %s (see docs/ocr-runbook.md)"
                     % (args.lang, ", ".join(defaults["models"])))
        if args.model != defaults["models"][0]:
            print("model %s is not installed; embedding with %s" % (defaults["models"][0], args.model))

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    units_path = args.units_out or os.path.join(src, units_name(args.lang))
    cites: dict[str, dict] = {}
    cite_path = os.path.join(src, "citations.jsonl")
    if os.path.exists(cite_path):
        with open(cite_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if "_meta" not in rec:
                    # A LIST, not one citation. The legacy-font resolver emits at
                    # most one per paragraph, but the transliteration resolver
                    # finds every verse in a paragraph and Bandginama runs three
                    # into one -- keyed singly, two of the three were overwritten
                    # and lost here, silently.
                    # the whole range of lines, and which scripture they are lines
                    # of: line_id alone lost both, and a Dasam Bani id read as a
                    # Guru Granth Sahib one is a wrong verse, silently
                    ids = rec.get("line_ids") or [rec["line_id"]]
                    cites.setdefault(rec["unit_id"], []).append(
                        {"shabad_id": rec["shabad_id"], "line_id": rec["line_id"],
                         "line_from": rec.get("line_from", min(ids)), "line_to": rec.get("line_to", max(ids)),
                         "source": rec.get("source") or "G", "role": "quotes", "page": rec.get("page"),
                         "ang": rec["ang"], "score": rec["score"],
                         "method": rec["method"], "span": rec["span"][:300]})
    # The links 35_link_scriptures.py made to the Dasam Bani, the Vaaran and the
    # Kabit Savaiye: a file of their own, so a resolver rewriting citations.jsonl
    # cannot take them. One that replaces a weak Granth link drops that link.
    links_path = os.path.join(src, "scripture-links.jsonl")
    if os.path.exists(links_path):
        added = dropped = 0
        with open(links_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if "_meta" in rec:
                    continue
                mine = cites.setdefault(rec["unit_id"], [])
                gone = rec.get("replaces")
                if gone:
                    keep = [c for c in mine if not (c["source"] == gone.get("source", "G") and c["shabad_id"] == gone.get("shabad_id"))]
                    dropped += len(mine) - len(keep)
                    mine[:] = keep
                mine.append({"shabad_id": rec["shabad_id"], "line_id": rec["line_id"],
                             "line_from": rec["line_from"], "line_to": rec["line_to"],
                             "source": rec["source"], "role": "quotes", "page": rec.get("page"),
                             "ang": rec["ang"], "score": rec["score"], "method": rec["method"],
                             "span": (rec.get("span") or "")[:300]})
                added += 1
        print("%d links to the other scriptures (%s), %d Granth links they replace dropped"
              % (added, os.path.basename(links_path), dropped))
    print("%d resolved citations in %d paragraphs"
          % (sum(len(v) for v in cites.values()), len(cites)))

    works, units = [], []
    # A folder holds more than its works: the citations, the units this writes,
    # the English of a translated work, 30_line_commentary.py's per-line teeka
    # (lines-<translator>.jsonl) and 35_link_scriptures.py's links
    # (scripture-links*.jsonl). None of them is a work, and reading one as a
    # work fails on its first record.
    not_works = ("units", "lines-", "scripture-links")
    paths = [p for p in sorted(glob.glob(os.path.join(src, "*.jsonl")))
             if not (os.path.basename(p) == "citations.jsonl" or os.path.basename(p).startswith(not_works)
                     or os.path.basename(p).endswith(".en.jsonl"))]
    if args.keep:
        if not os.path.exists(args.keep):
            sys.exit("--keep %s: no such database" % args.keep)
        works, units = kept_corpus(args.keep, {os.path.basename(p)[:-len(".jsonl")] for p in paths})
        for i, u in enumerate(units):
            u["unit_row"] = i
        print("kept %d works, %d units, as built in %s" % (len(works), len(units), args.keep))
    for path in paths:
        name = os.path.basename(path)
        meta, records = load_work(path)
        # a work ingested through a roster states no language: it is the
        # corpus's (the treatises are Punjabi, the essays English)
        language = meta.get("language") or args.lang
        translation = path[:-len(".jsonl")] + ".en.jsonl"
        if language != args.lang:
            if not (args.translations and os.path.exists(translation)):
                continue
            records, translated, dropped = apply_translations(records, translation)
            print("%s: %d paragraphs translated from %s, %d without a translation dropped"
                  % (meta["work"], translated, language, dropped))
            # its English is a machine's, not the author's own (works.original)
            meta = {**meta, "translated_from": language, "original": False}
        built = build_units(records, cites, args.target_tokens, args.hard_tokens)
        for u in built:
            u["unit_row"] = len(units)
            u["unit_id"] = "%s:%d:%d:%d" % (u["work"], u["part"] or 0, u["page"], u["para_no"])
            u["title"] = meta["title"]
            # a Gurmukhi title needs an English one beside it for a reader who
            # does not read the script; an English work has none and needs none
            u["title_en"] = meta.get("title_en")
            u["author"] = meta["author"]
            # the five works Bau Ji wrote in English may be quoted as written;
            # the essays are translations of his Punjabi and are summarised
            u["quote_policy"] = meta.get("quote_policy", "summarise")
            units.append(u)
        # the angs a work covers: what the manifest says when it says; else
        # the verses the work explains; else every verse it quotes (an essay
        # reaches wherever its quotations go)
        said = meta.get("angs") or []
        g = [c for u in built for c in u["cites"] if (c.get("source") or "G") == "G" and c.get("ang")]
        explained = [c["ang"] for c in g if c.get("role") == "explains"]
        reach = explained or [c["ang"] for c in g]
        works.append({**meta, "units": len(built),
                      "cites": sum(len(u["cites"]) for u in built),
                      "links": sum(len(u["cites"]) for u in built),
                      "ang_from": said[0] if said else (min(reach) if reach else None),
                      "ang_to": said[-1] if said else (max(reach) if reach else None)})
    print("%d works, %d units, %d citations attached"
          % (len(works), len(units), sum(len(u["cites"]) for u in units)))

    if not units:
        sys.exit("no works with language %r under %s" % (args.lang, src))
    with open(units_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": {"corpus": args.corpus, "language": args.lang, "works": works}},
                            ensure_ascii=False) + "\n")
        for u in units:
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    if args.units_only:
        print("  -> %s (units only: nothing embedded)" % units_path)
        return

    prof = profile(args.model)
    emb = Embedder(args.model, max_len=args.max_len)
    texts = [u["text"] for u in units]
    print("embedding %d units with %s (max_len %d)" % (len(texts), args.model, args.max_len))
    # one unit at a time, as a query is embedded: embed_long's default (lib/embedder.py BATCH_SIZE)
    vectors, chunked = embed_long(emb, texts, args.max_len)
    print("  %d units were longer than the window and were chunked" % chunked)

    index_dim = index_dim_for(len(units), int(vectors.shape[1]))
    if index_dim < OUT_DIM:
        print("  %d units: the index keeps %d dimensions, not %d" % (len(units), index_dim, OUT_DIM))
    pca = PCA(n_components=index_dim, svd_solver="full", random_state=0)
    reduced = pca.fit_transform(vectors.astype(np.float32))
    reduced /= np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-12)
    codes, scale = quantize(reduced.astype(np.float32))
    # how much the int8 round trip costs, the same check 02_reduce_quantize makes
    approx = codes.astype(np.float32) * scale[:, None]
    approx /= np.maximum(np.linalg.norm(approx, axis=1, keepdims=True), 1e-12)
    cos = np.sum(approx * reduced, axis=1)

    out_dir = args.out_dir or os.path.join(ARTIFACTS, "corpora", args.corpus)
    os.makedirs(out_dir, exist_ok=True)
    codes.tofile(os.path.join(out_dir, "units.i8"))
    scale.astype(np.float32).tofile(os.path.join(out_dir, "units.scale.f32"))
    np.ones(len(units), dtype=np.uint8).tofile(os.path.join(out_dir, "units.mask.u8"))
    pca.components_.astype(np.float32).tofile(os.path.join(out_dir, "pca.components.f32"))
    pca.mean_.astype(np.float32).tofile(os.path.join(out_dir, "pca.mean.f32"))

    licences: dict = {}
    for w in works:
        licences[str(w.get("licence") or "unknown")] = licences.get(str(w.get("licence") or "unknown"), 0) + 1
    manifest = {
        # the fields the server's registry reads of any index; `kind` is what
        # keeps this one out of the scripture routes, and `db` names the
        # database 15_build_writings_db.py writes for this corpus, relative
        # to this directory
        "kind": "documents", "index": args.corpus, "db": "../../" + db_name(args.corpus),
        "label": LABELS[args.lang], "label_pa": LABELS_PA[args.lang], "roles": ["text"],
        "order": 200 + sorted(LANG_DEFAULTS).index(args.lang),
        "corpus": args.corpus, "author": units[0]["author"] if units else None,
        "authors": sorted({u["author"] for u in units}),
        "licences": licences,
        # the model block uses the same field names an index manifest uses, so
        # query-encoder's encoderOptions() reads it unchanged
        "model": prof["model"], "model_dir": prof["model_dir"], "tokenizer": prof["tokenizer"],
        "pooling": prof["pooling"], "query_prefix": prof["query_prefix"], "doc_prefix": prof["doc_prefix"],
        "pad_token": prof["pad_token"], "pad_id": prof["pad_id"],
        "lowercase": bool(prof.get("lowercase", True)), "strip_accents": bool(prof.get("strip_accents", True)),
        "max_len": args.max_len, "embed_dim": int(vectors.shape[1]), "index_dim": index_dim,
        "units": len(units), "works": len(works), "chunked_units": int(chunked),
        "built_with": build_env(BATCH_SIZE),
        "citations": sum(len(u["cites"]) for u in units),
        "links": sum(len(u["cites"]) for u in units),
        "text_lang": args.lang, "query_scripts": defaults["query_scripts"],
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "quantization": {"type": "int8", "scale": "per-vector float32",
                         "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean())},
        "files": {},
    }
    for fn in ("units.i8", "units.scale.f32", "units.mask.u8", "pca.components.f32", "pca.mean.f32"):
        manifest["files"][fn] = os.path.getsize(os.path.join(out_dir, fn))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print("  explained variance %.3f | int8 cosine min %.4f mean %.4f"
          % (manifest["explained_variance"], cos.min(), cos.mean()))
    print("  -> %s  and  %s" % (out_dir, units_path))


if __name__ == "__main__":
    main()
