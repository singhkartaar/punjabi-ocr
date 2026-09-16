"""
Several engines' readings of one line, made into one.

Two or three engines read the same commentary line; each gets a few
characters wrong, rarely the same ones. Aligned grapheme by grapheme and
voted, the result is better than any single reading -- the ROVER result,
reproduced on printed books at 14-50% fewer errors. What matters as much as
the text is the AGREEMENT: the share of positions on which every engine that
read the line said the same thing. It is the ambiguity signal the routing uses,
because it comes from independent witnesses, where an engine's own confidence
does not (Tesseract's is 100 + 5 x certainty and sits at 0 on correctly read
Gurmukhi words often enough to be useless alone).

The vote is pivot-based: every other reading is aligned to the pivot engine's
reading with edit operations over grapheme clusters, giving one column per
pivot grapheme plus insertion columns between them. A column's value is the
one with the most weight, weight being the engine's prior (from the bake-off)
times its confidence for the line where it has one. Ties go to the pivot.

Lines are paired across engines by their boxes, not their text: two boxes on
the same page that overlap by half their height and a third of their width are
the same line, and an engine that cut one line in two contributes the two
halves joined.
"""
from __future__ import annotations
from collections import defaultdict

from lib.ocr_text import graphemes, normalise

V_OVERLAP = 0.5
H_OVERLAP = 0.3
DEFAULT_CONF = 0.8         # an engine with no confidence votes at this


def _overlap(a: list, b: list) -> tuple[float, float]:
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return iy / max(1, min(a[3] - a[1], b[3] - b[1])), ix / max(1, min(a[2] - a[0], b[2] - b[0]))


def align_boxes(pivot: list[dict], other: list[dict]) -> dict[int, list[dict]]:
    """{pivot index: [other lines overlapping it, left to right]}."""
    out: dict[int, list[dict]] = defaultdict(list)
    for i, p in enumerate(pivot):
        for o in other:
            v, h = _overlap(p["bbox"], o["bbox"])
            if v >= V_OVERLAP and h >= H_OVERLAP:
                out[i].append(o)
        out[i].sort(key=lambda ln: ln["bbox"][0])
    return out


def joined_text(lines: list[dict]) -> str:
    return " ".join(ln.get("text", "") for ln in lines).strip()


def _editops(a: list[str], b: list[str]):
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.editops(a, b)


def vote(candidates: list[dict]) -> dict:
    """
    @param candidates [{"engine", "text", "conf", "weight"}], the pivot first
    @returns {"text", "agreement", "conf", "n"}
    """
    cands = [c for c in candidates if (c.get("text") or "").strip()]
    if not cands:
        return {"text": "", "agreement": None, "conf": None, "n": 0}
    pivot = cands[0]
    p = graphemes(normalise(pivot["text"]))
    if len(cands) == 1:
        return {"text": normalise(pivot["text"]), "agreement": None, "conf": pivot.get("conf"), "n": 1}

    def w(c):
        return float(c.get("weight", 1.0)) * float(c.get("conf") if c.get("conf") is not None else DEFAULT_CONF)

    # columns[i] -> {grapheme or "": weight}; inserts[i] -> {inserted string: weight} before pivot i
    columns: list[dict] = [defaultdict(float) for _ in p]
    inserts: list[dict] = [defaultdict(float) for _ in range(len(p) + 1)]
    seen_at: list[set] = [set() for _ in p]              # engines that voted at column i
    for c in cands:
        g = graphemes(normalise(c["text"]))
        wc = w(c)
        if c is pivot:
            for i, x in enumerate(p):
                columns[i][x] += wc + 1e-6              # the tie-break
                seen_at[i].add(c["engine"])
            continue
        ops = _editops(p, g)
        # replay: start with identity, apply ops to know each pivot position's fate
        fate: dict[int, str | None] = {i: p[i] for i in range(len(p))}
        ins: dict[int, list[str]] = defaultdict(list)
        for op in ops:
            if op.tag == "replace":
                fate[op.src_pos] = g[op.dest_pos]
            elif op.tag == "delete":
                fate[op.src_pos] = ""
            elif op.tag == "insert":
                ins[op.src_pos].append(g[op.dest_pos])
        for i in range(len(p)):
            columns[i][fate[i]] += wc
            seen_at[i].add(c["engine"])
        for i, chars in ins.items():
            inserts[i]["".join(chars)] += wc
    total_w = sum(w(c) for c in cands)
    out: list[str] = []
    agree = 0
    for i in range(len(p) + 1):
        if inserts[i]:
            s, ws = max(inserts[i].items(), key=lambda kv: kv[1])
            if ws > total_w / 2.0:                        # a majority inserted it
                out.append(s)
        if i < len(p):
            best, _ = max(columns[i].items(), key=lambda kv: kv[1])
            out.append(best)
            if len(columns[i]) == 1 and len(seen_at[i]) == len(cands):
                agree += 1
    text = normalise("".join(out))
    confs = [c["conf"] for c in cands if c.get("conf") is not None]
    return {"text": text, "agreement": round(agree / max(len(p), 1), 3),
            "conf": round(sum(confs) / len(confs), 3) if confs else None, "n": len(cands)}


def map_positions(src: list[str], dst: list[str]) -> list[int]:
    """
    For every boundary 0..len(src) in src, the corresponding boundary in dst
    under the edit alignment. Used to cut a block's text where the pivot's
    lines break.
    """
    from rapidfuzz.distance import Levenshtein
    out = [0] * (len(src) + 1)
    for op in Levenshtein.opcodes(src, dst):
        for b in range(op.src_start, op.src_end + 1):
            if op.tag in ("equal", "replace"):
                out[b] = min(op.dest_end, op.dest_start + (b - op.src_start))
            elif op.tag == "delete":
                out[b] = op.dest_start
            else:                                         # insert: src_start == src_end
                out[b] = op.dest_end if b == op.src_end else op.dest_start
    out[len(src)] = len(dst)
    return out


def segment_block(block: dict, pivot_lines: list[dict]) -> list[dict]:
    """
    One engine's paragraph block cut into the pivot's lines.

    Surya 2 and the VLM engines return a paragraph as one box and one text.
    The vote needs lines, so the block's graphemes are aligned to the
    concatenated graphemes of the pivot lines its box covers, and cut at the
    pivot's line boundaries. Each piece keeps the pivot line's box.
    """
    inside = [p for p in pivot_lines
              if _overlap(p["bbox"], block["bbox"])[0] >= 0.5 and _overlap(p["bbox"], block["bbox"])[1] >= 0.3]
    inside.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))
    if len(inside) <= 1:
        return [dict(block)]
    pg = []
    bounds = [0]
    for p in inside:
        g = graphemes(normalise(p.get("text", "")))
        pg.extend(g + [" "])
        bounds.append(len(pg))
    bg = graphemes(normalise(block.get("text", "")))
    pos = map_positions(pg, bg)
    out = []
    for i, p in enumerate(inside):
        a, b = pos[bounds[i]], pos[bounds[i + 1]]
        piece = "".join(bg[a:b]).strip()
        out.append({**block, "bbox": list(p["bbox"]), "text": piece, "segmented": True})
    return out


def is_block(line: dict, typical_h: float) -> bool:
    """A box more than twice a typical line tall is a paragraph block."""
    return (line["bbox"][3] - line["bbox"][1]) > 2.0 * typical_h and "\n" not in (line.get("text") or "")
