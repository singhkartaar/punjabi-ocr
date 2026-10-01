"""
Link a translation of a bani to the shabads it translates, paragraph by paragraph.

  29_link_translations.py --src data/barusahib --work teja-singh --part 1 \
      --pages 4-48 --shabads 1-39                 # Japji Sahib
  29_link_translations.py --src data/barusahib --work asa-di-var \
      --pages 49-112 --shabads 1685-1767          # Asa di Var

Sant Teja Singh's Japji Sahib and Asa di Var ARE Gurbani, in his English, and
print no ang beside it: nothing else in this pipeline can cite them, because
every other resolver needs the ang, the Gurmukhi, or a romanisation of it. What
they do have is ORDER. A translation walks through the bani pauri by pauri, so
the question is not "which of 60,000 lines is this?" but "where in this one
bani's 39 (or 83) shabads has the book got to?".

So each paragraph is scored against every shabad of the bani by the English
the corpus already holds for it (Manmohan Singh, Sant Singh Khalsa and the SSK
translation, as TF-IDF over stemmed content words, a paragraph read with its
neighbours either side), the score centred on that paragraph's mean, and the
paragraphs are then segmented IN ORDER: each one stays on the shabad the one
before it was on or moves forward, never back, and moving costs PENALTY, so
noise in one paragraph cannot make the path jump. A paragraph is cited to its
segment's shabad where it reads above its own average for that shabad, at the
line of that shabad it reads most like.

Measured by the book's own words: "In the next four Pauris (12, 13, 14, and
15)" lands on pauri 12, "the next four Pauries (28 to 31)" on 28, "This solok
gives the gist of the Japji Sahib" on the closing salok, and Asa di Var's
"STANZA IX", "STANZA XVIII" and "STANZA XX" on those pauris. Pauris a book
explains together share a segment; the citation then names the first.

Only this step's own earlier citations are replaced (how == "translation");
what every other resolver wrote stays, as 15-resolve-translit.js --append
keeps it, and a verse already cited for a paragraph is not cited twice.
"""
import argparse
import json
import math
import os
import re
import sqlite3
from collections import Counter

from lib.paths import CORPUS_DB, ROOT

TRANSLATORS = ("ms", "bdb", "ssk")
PENALTY = 0.15
STOP = set("""a an the and or of to in on at by for with from as is are was were be been being it its this that these
those which who whom whose what when where how why he she they them his her their him i me my we us our you your thou
thee thy thine ye not no nor but so if then than there here all any each every some such one also very can could shall
should will would may might must do does did done has have had having o oh unto upon into within without through about
over under again ever only own same other s t just even""".split())
SUFFIXES = ("ings", "ing", "eth", "est", "ies", "ied", "ed", "es", "s")


def words(text: str) -> list[str]:
    """Content words, crudely stemmed: what two translations of one line share."""
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP or len(w) < 3:
            continue
        for suf in SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class Space:
    """TF-IDF over the shabads of one bani; idf from those shabads alone."""

    def __init__(self, docs: dict):
        df = Counter()
        for d in docs.values():
            df.update(set(d))
        self.n = len(docs)
        self.idf = {w: math.log((self.n + 1) / (c + 1)) + 1 for w, c in df.items()}

    def vec(self, ws: list[str]) -> dict:
        c = Counter(ws)
        unseen = math.log(self.n + 1) + 1
        v = {w: (1 + math.log(k)) * self.idf.get(w, unseen) for w, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}


def cos(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def segment(sim: list[list[float]], penalty: float) -> list[int]:
    """Each row to a column, columns non-decreasing; a change of column costs `penalty`."""
    n, m = len(sim), len(sim[0])
    best = [row[:] for row in sim]
    back = [[0] * m for _ in range(n)]
    for i in range(1, n):
        run, arg = -1e18, 0
        for j in range(m):
            stay, move = best[i - 1][j], run - penalty
            if stay >= move:
                best[i][j], back[i][j] = stay + sim[i][j], j
            else:
                best[i][j], back[i][j] = move + sim[i][j], arg
            if best[i - 1][j] > run:
                run, arg = best[i - 1][j], j
    j = max(range(m), key=lambda k: best[-1][k])
    path = [j]
    for i in range(n - 1, 0, -1):
        j = back[i][j]
        path.append(j)
    return path[::-1]


def span(arg: str) -> tuple[int, int]:
    lo, _, hi = arg.partition("-")
    return int(lo), int(hi or lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of <work>.jsonl + citations.jsonl")
    ap.add_argument("--work", required=True)
    ap.add_argument("--part", type=int, default=None, help="only this part of the work")
    ap.add_argument("--pages", required=True, help="the translation's pages, e.g. 49-112")
    ap.add_argument("--shabads", required=True, help="the bani's shabad_ids in the corpus, e.g. 1685-1767")
    ap.add_argument("--penalty", type=float, default=PENALTY)
    ap.add_argument("--print", dest="show", action="store_true", help="print each segment")
    args = ap.parse_args()
    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    plo, phi = span(args.pages)
    slo, shi = span(args.shabads)

    con = sqlite3.connect(CORPUS_DB)
    marks = ",".join("?" * len(TRANSLATORS))
    rows = con.execute(
        "SELECT l.shabad_id, l.line_id, l.ang, l.kind, t.text FROM lines l "
        "JOIN translations t ON t.line_id = l.line_id "
        "WHERE l.shabad_id BETWEEN ? AND ? AND t.translator IN (%s) ORDER BY l.line_id" % marks,
        (slo, shi, *TRANSLATORS)).fetchall()
    docs, lines = {}, {}
    for sid, lid, ang, kind, text in rows:
        ws = words(text)
        docs.setdefault(sid, []).extend(ws)
        ln = lines.setdefault(sid, {}).setdefault(lid, {"ang": ang, "kind": kind, "words": []})
        ln["words"].extend(ws)
    if not docs:
        raise SystemExit("no translated lines for shabads %d-%d in %s" % (slo, shi, CORPUS_DB))
    ids = sorted(docs)
    space = Space(docs)
    dv = {s: space.vec(docs[s]) for s in ids}
    lv = {s: {lid: space.vec(l["words"]) for lid, l in lines[s].items()} for s in ids}

    path_in = os.path.join(src, args.work + ".jsonl")
    recs = [json.loads(l) for l in open(path_in, encoding="utf-8").read().splitlines()[1:] if l]
    recs = [r for r in recs if plo <= r["page"] <= phi and (args.part is None or r.get("part") == args.part)]
    if not recs:
        raise SystemExit("no paragraphs of %s on pages %d-%d" % (args.work, plo, phi))
    toks = [words(r["text"]) for r in recs]
    sim = []
    for i in range(len(recs)):
        # read with a neighbour either side, itself counted twice
        ctx = toks[max(0, i - 1)] + toks[i] + toks[i] + toks[min(len(toks) - 1, i + 1)]
        v = space.vec(ctx)
        row = [cos(v, dv[s]) for s in ids]
        mu = sum(row) / len(row)
        sim.append([x - mu for x in row])
    path = segment(sim, args.penalty)

    citations, segs = [], []
    for i, (r, j) in enumerate(zip(recs, path)):
        sid = ids[j]
        if not segs or segs[-1]["shabad_id"] != sid:
            segs.append({"shabad_id": sid, "from": i, "to": i, "page": r["page"]})
        segs[-1]["to"] = i
        if sim[i][j] <= 0:
            continue            # reads below its own average here: not this verse's translation
        own = space.vec(toks[i])
        cand = [(cos(own, v), lid) for lid, v in lv[sid].items() if lines[sid][lid]["kind"] in ("line", "rahao")]
        score, lid = max(cand) if cand else (0.0, min(lines[sid]))
        if score <= 0:
            lid = min(l for l in lines[sid] if lines[sid][l]["kind"] in ("line", "rahao")) if cand else lid
        citations.append({
            "unit_id": r["unit_id"], "work": r["work"], "part": r.get("part"), "page": r["page"],
            "para_no": r["para_no"], "ang": lines[sid][lid]["ang"], "how": "translation",
            "span": r["text"][:300], "shabad_id": sid, "line_id": lid, "line_to": lid,
            "score": round(sim[i][j], 3), "margin": 0, "method": "aligned",
        })

    out = os.path.join(src, "citations.jsonl")
    kept, cited = [], set()
    meta = {}
    if os.path.exists(out):
        lines_out = open(out, encoding="utf-8").read().splitlines()
        meta = json.loads(lines_out[0]).get("_meta", {}) if lines_out else {}
        for l in lines_out[1:]:
            if not l:
                continue
            c = json.loads(l)
            # this step's own earlier output for THIS work is replaced
            if c.get("how") == "translation" and c.get("work") == args.work and \
                    (args.part is None or c.get("part") == args.part):
                continue
            kept.append(c)
            cited.add((c["unit_id"], c["shabad_id"]))
    mine = [c for c in citations if (c["unit_id"], c["shabad_id"]) not in cited]
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for c in kept + mine:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    covered = len({s["shabad_id"] for s in segs})
    print("%s%s pages %d-%d: %d paragraphs, %d segments over %d of %d shabads, %d citations"
          % (args.work, "" if args.part is None else " part %d" % args.part, plo, phi,
             len(recs), len(segs), covered, len(ids), len(mine)))
    if args.show:
        for s in segs:
            print("  shabad %5d  p%-4d paragraphs %4d-%-4d  %s" % (
                s["shabad_id"], s["page"], s["from"], s["to"], recs[s["from"]]["text"][:70]))


if __name__ == "__main__":
    main()
