"""
Which translation engine to trust, measured on a few dozen paragraphs against a paid reference.

  28_translate_bench.py --work santhya --sample 30 --engines sarvam,indictrans2 --reference vertex --budget-usd 1
  28_translate_bench.py --work santhya --engines existing,indictrans2 --reference cached      # no paid call
  28_translate_bench.py --work santhya --reference vertex --dry-run                          # the cost, nothing sent

The Vertex run that translated the Darpan cost $13K; translating a book with
it is not on the table. Translating THIRTY paragraphs with it is cents, and
that is what it is for here: a reference the local engines are scored against
with chrF and BLEU (sacrebleu), plus the length ratio that catches an engine
that pads or truncates. The sample and the reference are written to
data/raw/mt-bench-<work>.json, so a later run (--reference cached) scores a
new engine without paying again.

Engines: sarvam and indictrans2 as in 26_translate_writings.py; "existing"
reads the translations already in data/writings/<work>.en.jsonl (whatever
engine made them), so a finished run is scored without repeating it.

The sample is stratified by length (short, medium, long thirds) with a fixed
seed, so two runs compare the same paragraphs. The five lowest-scoring pairs
per engine are printed: a number says how far, the pairs say why.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.mt_engines import read_jsonl
from lib.paths import OCR_COSTS, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tw = import_module("26_translate_writings")
WRITINGS = tw.WRITINGS
RAW = os.path.join(ROOT, "data", "raw")


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    """n records spread over the length thirds, in a fixed order."""
    if n <= 0 or n >= len(records):
        return list(records)
    by_len = sorted(records, key=lambda r: len(r["text"]))
    third = max(1, len(by_len) // 3)
    parts = [by_len[:third], by_len[third:2 * third], by_len[2 * third:]]
    rng = random.Random(seed)
    out: list[dict] = []
    for i, part in enumerate(parts):
        want = n // 3 + (1 if i < n % 3 else 0)
        out.extend(rng.sample(part, min(want, len(part))))
    return sorted(out, key=lambda r: r["unit_id"])


def score(hyps: list[str], refs: list[str], srcs: list[str]) -> dict:
    """chrF and BLEU over the paired lists, per-pair chrF, and the length ratio."""
    from sacrebleu.metrics import BLEU, CHRF
    chrf, bleu = CHRF(), BLEU()
    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return {"chrf": 0.0, "bleu": 0.0, "ratio": 0.0, "scored": 0, "per_pair": []}
    hs, rs = [p[0] for p in pairs], [p[1] for p in pairs]
    per_pair = [round(chrf.sentence_score(h, [r]).score, 1) for h, r in pairs]
    ratio = sum(len(h) for h in hyps if h) / max(1, sum(len(s) for s, h in zip(srcs, hyps) if h))
    return {"chrf": round(chrf.corpus_score(hs, [rs]).score, 1), "bleu": round(bleu.corpus_score(hs, [rs]).score, 1),
            "ratio": round(ratio, 2), "scored": len(pairs), "per_pair": per_pair}


def existing_translations(work: str) -> dict[str, str]:
    path = os.path.join(WRITINGS, work + ".en.jsonl")
    return {r["unit_id"]: r["en"] for r in read_jsonl(path) if r.get("unit_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engines", default="sarvam,indictrans2",
                    help="comma list of sarvam, indictrans2, madlad, nllb, llama, existing")
    ap.add_argument("--path", action="append", default=[], metavar="ENGINE=PATH",
                    help="weights or URL for an engine, e.g. sarvam=../../vendor/models/sarvam-translate, "
                         "llama=http://127.0.0.1:8081")
    ap.add_argument("--as", dest="labels", action="append", default=[], metavar="ENGINE=LABEL",
                    help="the row name to record, e.g. llama=gemma3-12b (one llama-server model per run)")
    ap.add_argument("--reference", default="vertex", choices=["vertex", "cached"])
    ap.add_argument("--budget-usd", type=float, default=1.0)
    ap.add_argument("--src-lang", choices=sorted(tw.LANG_NAME))
    ap.add_argument("--styles", default="body")
    ap.add_argument("--sarvam-path", default=tw.SARVAM)
    ap.add_argument("--indictrans2-path", default=tw.INDICTRANS2)
    ap.add_argument("--dry-run", action="store_true", help="print the sample and the reference cost, run nothing")
    args = ap.parse_args()

    src_path = os.path.join(WRITINGS, args.work + ".jsonl")
    if not os.path.exists(src_path):
        sys.exit("no %s (12_ingest_writings.py)" % src_path)
    lang = args.src_lang or tw.work_language(src_path) or "pa"
    lang_name = tw.LANG_NAME[lang]
    out_path = os.path.join(RAW, "mt-bench-%s.json" % args.work)
    cached = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else None

    if cached and args.reference == "cached":
        sample = cached["sample"]
    else:
        styles = set(args.styles.split(","))
        records = [r for r in read_jsonl(src_path) if r.get("style") in styles and r.get("lang", lang) != "en"
                   and len(r["text"]) >= 40 and "॥" not in r["text"]]      # prose, not a quoted verse
        sample = [{"unit_id": r["unit_id"], "text": " ".join(r["text"].split())}
                  for r in sample_records(records, args.sample, args.seed)]
        # an earlier reference for the same paragraph is kept: it cost money
        old = {s["unit_id"]: s.get("reference") for s in (cached or {}).get("sample", [])}
        for s in sample:
            if old.get(s["unit_id"]):
                s["reference"] = old[s["unit_id"]]
    need = [s for s in sample if not s.get("reference")]
    chars = sum(len(s["text"]) for s in need)
    est = round(chars / 1000.0 * tw.VERTEX_USD_PER_1K_CHARS, 4)
    print("%s (%s): %d paragraphs in the sample, %d without a reference (%d chars, est. $%.4f on Vertex)"
          % (args.work, lang_name, len(sample), len(need), chars, est))
    if args.dry_run:
        for s in sample[:5]:
            print("  %s: %s" % (s["unit_id"], s["text"][:100]))
        return
    if need and args.reference == "cached":
        # a reference the checks rejected stays missing; those paragraphs are
        # simply not scored, the same as when the reference was fetched
        if len(need) == len(sample):
            sys.exit("no cached reference for any paragraph; run with --reference vertex")
        print("  %d paragraph(s) without a reference are left out of the score" % len(need))

    if need and args.reference == "vertex":
        ref = tw.make_engine("vertex", None, lang, args.budget_usd)
        for i, s in enumerate(need, 1):
            en = ref.translate(s["text"], lang_name)
            if tw.check(en, s["text"], lang):
                print("  reference rejected for %s: %s" % (s["unit_id"], tw.check(en, s["text"], lang)))
                continue
            s["reference"] = " ".join(en.split())
            if i % 10 == 0:
                print("  reference %d/%d" % (i, len(need)))
        reference = {"engine": ref.name, "model": ref.model_desc()}
    else:
        reference = (cached or {}).get("reference") or {"engine": "cached"}

    result = {"work": args.work, "language": lang, "seed": args.seed, "reference": reference,
              "sample": sample, "engines": (cached or {}).get("engines", {})}
    texts = [s["text"] for s in sample]
    refs = [s.get("reference", "") for s in sample]
    paths = dict(kv.split("=", 1) for kv in args.path)
    paths.setdefault("sarvam", args.sarvam_path)
    paths.setdefault("indictrans2", args.indictrans2_path)
    labels = dict(kv.split("=", 1) for kv in args.labels)
    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        t0 = time.time()
        row = labels.get(name, name)
        if name == "existing":
            have = existing_translations(args.work)
            hyps = [have.get(s["unit_id"], "") for s in sample]
            desc = "data/writings/%s.en.jsonl" % args.work
        else:
            engine = tw.make_engine(name, paths.get(name), lang)
            desc = engine.model_desc()
            hyps = []
            for b in range(0, len(texts), 8):
                hyps.extend(engine.translate_many(texts[b:b + 8], lang_name))
            del engine
        reasons = [tw.check(h, s, lang) for h, s in zip(hyps, texts)]
        rejected = sum(1 for r in reasons if r)
        for s, h, why in zip(sample, hyps, reasons):
            if why:
                print("   rejected %s: %s\n     %s: %s" % (s["unit_id"], why, row, (h or "")[:160]))
        hyps = [h if not why else "" for h, why in zip(hyps, reasons)]
        sc = score(hyps, refs, texts)
        result["engines"][row] = {"model": desc, "chrf": sc["chrf"], "bleu": sc["bleu"], "ratio": sc["ratio"],
                                   "scored": sc["scored"], "rejected": rejected,
                                   "seconds": round(time.time() - t0, 1),
                                   "translations": {s["unit_id"]: h for s, h in zip(sample, hyps)},
                                   "per_pair": sc["per_pair"]}
        print("%-12s chrF %5.1f  BLEU %5.1f  len x%.2f  scored %d  rejected %d  %.0fs  (%s)"
              % (row, sc["chrf"], sc["bleu"], sc["ratio"], sc["scored"], rejected, time.time() - t0, desc))
        scored = [(s, h) for s, h in zip(sample, hyps) if h and s.get("reference")]   # the pairs score() saw
        worst = sorted(zip(sc["per_pair"], [s for s, _ in scored], [h for _, h in scored]), key=lambda t: t[0])[:5]
        for pp, s, h in worst:
            print("   chrF %.0f %s\n     %s: %s\n     %s: %s\n     ref: %s"
                  % (pp, s["unit_id"], lang.upper(), s["text"][:140], row, h[:160], s["reference"][:160]))

    os.makedirs(RAW, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print("-> %s" % os.path.relpath(out_path, ROOT))


if __name__ == "__main__":
    main()
