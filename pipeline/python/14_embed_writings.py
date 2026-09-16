"""
Build the writings corpus: retrieval units, their vectors, their citations.

  14_embed_writings.py
  14_embed_writings.py --corpus writings-en --target-tokens 60

Writes data/writings/units.jsonl (what a reader is shown and what the server
serves) and artifacts/<corpus>/ (what retrieval reads).

Two decisions are made here and both are load-bearing.

**A source paragraph is not a retrieval unit.** Nearly half of this author's
paragraphs are ten tokens or fewer -- he writes in cascading indented lists
("studied or cause to be studied / learnt or taught / taken or given") -- and a
unit reading "recognise" retrieves nothing and means nothing. Paragraphs are
merged forward to a target size, never across a work, a part or a heading. The
merge also repairs the 4.4% of paragraphs the reader splits mid-sentence, which
is why it over-segments there rather than under.

**The quoted Gurbani is not part of the unit's text.** A quotation is Gurbani's
words, already searchable through the `en` index over the whole Granth; folding
it in would make this corpus retrieve for verses instead of for what Bau Ji says
about them, and the two indexes would stop doing different jobs. The quotation
becomes a CITATION on the unit that leads into it, carrying the shabad id, so an
answer can quote the exposition and show the verse beneath it.

The corpus is an index directory like any other, artifacts/<corpus>/, whose
manifest says `"kind": "documents"`: its rows are passages, not lines of the
Granth, and everything that scans artifacts/ (the server's registry,
08_annotate_manifest.py, test_pipeline.py, the integration test, the pack
builder) branches on that field rather than on where the directory sits. The
server keeps documents indexes in a map of their own and serves them through
/api/documents; 15_build_writings_db.py puts the text beside the vectors as
units.sqlite, so the directory is the whole corpus.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.documents import embed_long
from lib.embedder import Embedder
from lib.models import first_available, profile
from lib.paths import ARTIFACTS, ROOT
from lib.quantize import quantize

WRITINGS = os.path.join(ROOT, "data", "writings")
UNITS = os.path.join(WRITINGS, "units.jsonl")
OUT_DIM = 256
ENDS_SENTENCE = tuple(".!?:;–—”’\")।॥")          # Latin punctuation, or a danda
# One corpus per language: the English essays embed with the English model,
# a Punjabi book with the Gurbani-tuned multilingual one, and each corpus
# keeps its own units file and unit_row space.
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


def build_units(records: list[dict], citations: dict, target: int, hard: int) -> list[dict]:
    """
    Paragraphs merged into retrieval units, with the verses they lead into.

    A unit closes once it is big enough AND ends on a full stop, so a merge
    never leaves a sentence hanging; it closes regardless at `hard` tokens, at a
    heading, and at a part boundary.
    """
    units: list[dict] = []
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur and cur["words"] >= 3:
            cur["text"] = " ".join(cur["text"]).strip()
            if cur["text_src"]:
                cur["text_src"] = " ".join(cur["text_src"]).strip()
            else:
                del cur["text_src"]
            del cur["words"]
            units.append(cur)
        cur = None

    def fresh(rec: dict) -> dict:
        return {"work": rec["work"], "part": rec["part"], "page": rec["page"],
                "para_no": rec["para_no"], "marker": rec.get("marker"),
                "text": [], "text_src": [], "words": 0, "cites": []}

    for rec in records:
        if rec["style"] == "quote":
            # the verse belongs to the prose that introduced it
            cite = citations.get(rec["unit_id"])
            if cite and cur:
                cur["cites"].append(cite)
            elif cite:
                units and units[-1]["cites"].append(cite)
            continue
        if rec["style"] == "heading":
            flush()
        if cur is None:
            cur = fresh(rec)
        elif (cur["work"], cur["part"]) != (rec["work"], rec["part"]):
            flush()
            cur = fresh(rec)
        cur["text"].append(rec["text"])
        if rec.get("text_src"):
            cur["text_src"].append(rec["text_src"])
        cur["words"] += len(rec["text"].split())
        whole = rec["text"].rstrip().endswith(ENDS_SENTENCE)
        if cur["words"] >= hard or (cur["words"] >= target and whole):
            flush()
    flush()
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=sorted(LANG_DEFAULTS),
                    help="which works to embed (their _meta.language); sets the corpus and model defaults")
    ap.add_argument("--corpus", help="default: writings-<lang>")
    ap.add_argument("--model", help="default: the language's model (LANG_DEFAULTS)")
    ap.add_argument("--units-out", help="default: data/writings/units.jsonl for en, units-<lang>.jsonl otherwise")
    ap.add_argument("--max-len", type=int, default=256, help="token window for a unit")
    ap.add_argument("--target-tokens", type=int, default=60, help="words a unit aims for")
    ap.add_argument("--hard-tokens", type=int, default=220, help="words a unit never exceeds")
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
    units_path = args.units_out or (UNITS if args.lang == "en" else os.path.join(WRITINGS, "units-%s.jsonl" % args.lang))

    cites: dict[str, dict] = {}
    cite_path = os.path.join(WRITINGS, "citations.jsonl")
    if os.path.exists(cite_path):
        with open(cite_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if "_meta" not in rec:
                    cites[rec["unit_id"]] = {"shabad_id": rec["shabad_id"], "line_id": rec["line_id"],
                                             "ang": rec["ang"], "score": rec["score"],
                                             "method": rec["method"], "span": rec["span"][:300]}
    print("%d resolved citations" % len(cites))

    works, units = [], []
    for path in sorted(glob.glob(os.path.join(WRITINGS, "*.jsonl"))):
        name = os.path.basename(path)
        if name == "citations.jsonl" or name.startswith("units") or name.endswith(".en.jsonl"):
            continue
        meta, records = load_work(path)
        language = meta.get("language") or "en"
        translation = path[:-len(".jsonl")] + ".en.jsonl"
        if language != args.lang:
            if not (args.translations and os.path.exists(translation)):
                continue
            records, translated, dropped = apply_translations(records, translation)
            print("%s: %d paragraphs translated from %s, %d without a translation dropped"
                  % (meta["work"], translated, language, dropped))
            meta = {**meta, "translated_from": language}
        built = build_units(records, cites, args.target_tokens, args.hard_tokens)
        for u in built:
            u["unit_row"] = len(units)
            u["unit_id"] = "%s:%d:%d:%d" % (u["work"], u["part"] or 0, u["page"], u["para_no"])
            u["title"] = meta["title"]
            u["author"] = meta["author"]
            # the five works Bau Ji wrote in English may be quoted as written;
            # the essays are translations of his Punjabi and are summarised
            u["quote_policy"] = meta.get("quote_policy", "summarise")
            units.append(u)
        works.append({**meta, "units": len(built),
                      "cites": sum(len(u["cites"]) for u in built)})
    print("%d works, %d units, %d citations attached"
          % (len(works), len(units), sum(len(u["cites"]) for u in units)))

    if not units:
        sys.exit("no works with language %r under %s" % (args.lang, WRITINGS))
    with open(units_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": {"corpus": args.corpus, "language": args.lang, "works": works}},
                            ensure_ascii=False) + "\n")
        for u in units:
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")

    prof = profile(args.model)
    emb = Embedder(args.model, max_len=args.max_len)
    texts = [u["text"] for u in units]
    print("embedding %d units with %s (max_len %d)" % (len(texts), args.model, args.max_len))
    vectors, chunked = embed_long(emb, texts, args.max_len)
    print("  %d units were longer than the window and were chunked" % chunked)

    pca = PCA(n_components=OUT_DIM, svd_solver="full", random_state=0)
    reduced = pca.fit_transform(vectors.astype(np.float32))
    reduced /= np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-12)
    codes, scale = quantize(reduced.astype(np.float32))
    # how much the int8 round trip costs, the same check 02_reduce_quantize makes
    approx = codes.astype(np.float32) * scale[:, None]
    approx /= np.maximum(np.linalg.norm(approx, axis=1, keepdims=True), 1e-12)
    cos = np.sum(approx * reduced, axis=1)

    out_dir = os.path.join(ARTIFACTS, args.corpus)
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
        # keeps this one out of the scripture routes
        "kind": "documents", "index": args.corpus, "db": "units.sqlite",
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
        "max_len": args.max_len, "embed_dim": int(vectors.shape[1]), "index_dim": OUT_DIM,
        "units": len(units), "works": len(works), "chunked_units": int(chunked),
        "citations": sum(len(u["cites"]) for u in units),
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
