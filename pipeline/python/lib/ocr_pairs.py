"""
A two-column page read as bands: full-width prose, and verse paired with the
explanation printed beside it.

The Santhya's page is not two columns of text. It is prose across the whole
page (the previous pauri's arth, a preface), then a row that says ਮੂਲ | ਅਰਥ,
then the paired band -- verse chunks down the narrow left column, each with
its arth beside it on the right -- then prose across the page again, and the
footnotes. Reading the left column whole and then the right (the older
`columns` layout) ran the verse chunks together and put every arth after all
of them, and a prose line the merge had cut at the gutter came back as two
half lines far apart.

So the page is cut into bands first. Lines at the same height in both columns
are a ROW: two halves of the same style (prose and prose, heading and
heading) are one line the gutter split, and belong to a full-width band; a
verse half beside a prose half is a paired row. A run of paired rows is a
paired band, and inside it every prose paragraph takes the verse lines
printed beside it, by vertical overlap, as the lines it explains. One paired
row on its own among full-width rows is a full-width line with a bold lead
word ("ਪ੍ਰਾਕਥਨ- ..."), not a band.

Thresholds are multiples of the page's line height, never pixels: measured
on the Santhya at 300 dpi (page 70: rows split at the gutter agree in y to
within 3 px of a 55 px line; the arth paragraphs sit within half a line of
the verse they explain), and written down beside each constant.
"""
from __future__ import annotations
import re
import statistics

VERSE_KINDS = ("gurbani", "gurbani-unmatched", "heading")
ROW_TOL = 0.5        # of the line height: two lines are a row when their vertical overlap is this much of the smaller
NEAR = 3.0           # of the line height: a prose paragraph with no verse overlapping it takes verse this far above
VERSE_SHARE = 0.6    # share of a band's left lines that must be verse-like, and of its right lines prose, to pair
MIN_PAIRED_ROWS = 2  # paired rows a band needs; one alone is a lead word beside its sentence


WIDE_TOL = 0.5       # of the line height: a line reaching this far past the gutter on both sides is a full-width line


def column_of(ln: dict, gutter: float) -> int:
    """0 or 1 by where the line's centre falls: a bar the engine read out of the rule can put its left edge over the gutter."""
    return 0 if (ln["bbox"][0] + ln["bbox"][2]) / 2.0 < gutter else 1


def is_wide(ln: dict, gutter: float, h: float) -> bool:
    """A line printed across both columns: prose above or below the paired band, a title, the ਮੂਲ|ਅਰਥ row read whole."""
    return ln["bbox"][0] < gutter - WIDE_TOL * h and ln["bbox"][2] > gutter + WIDE_TOL * h


def is_verse_line(ln: dict) -> bool:
    """
    A line of the verse column: matched or unmatched Gurbani, or a short bold
    line the merge called a heading -- which in the verse column is a wrapped
    verse fragment of six words or fewer with no danda on it ("ਗਾਵੈ ਕੋ ਤਾਣ
    ਹੋਵੈ ਕਿਸੈ"), the merge's classifier not knowing the column.
    """
    return ln.get("kind") in VERSE_KINDS


def typical_height(lines: list[dict], default: float = 40.0) -> float:
    heights = sorted(ln["bbox"][3] - ln["bbox"][1] for ln in lines if ln.get("text", "").strip())
    return float(heights[len(heights) // 2]) if heights else default


def _v_overlap(a: list, b: list) -> float:
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iy / max(1.0, float(min(a[3] - a[1], b[3] - b[1])))


def rows_of(col0: list[dict], col1: list[dict], h: float) -> list[dict]:
    """
    The page's rows, top to bottom: {"left", "right"} where a left and a right
    line sit at the same height (each the other's best overlap), else one of
    them alone. `split` marks from the merge are believed first.
    """
    left = sorted(col0, key=lambda l: l["bbox"][1])
    right = sorted(col1, key=lambda l: l["bbox"][1])
    best_r: dict[int, tuple[float, int]] = {}
    best_l: dict[int, tuple[float, int]] = {}
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            v = _v_overlap(a["bbox"], b["bbox"])
            if v >= ROW_TOL:
                if v > best_r.get(i, (0.0, -1))[0]:
                    best_r[i] = (v, j)
                if v > best_l.get(j, (0.0, -1))[0]:
                    best_l[j] = (v, i)
    rows: list[dict] = []
    used_r: set[int] = set()
    for i, a in enumerate(left):
        j = best_r.get(i, (0.0, -1))[1]
        if j >= 0 and best_l.get(j, (0.0, -1))[1] == i:
            rows.append({"left": a, "right": right[j]})
            used_r.add(j)
        else:
            rows.append({"left": a, "right": None})
    for j, b in enumerate(right):
        if j not in used_r:
            rows.append({"left": None, "right": b})
    rows.sort(key=lambda r: min(x["bbox"][1] for x in (r["left"], r["right"]) if x))
    for r in rows:
        a, b = r["left"], r["right"]
        if a and b:
            r["kind"] = "paired" if is_verse_line(a) != is_verse_line(b) else "full"
        else:
            r["kind"] = "single"
    return rows


def join_row(a: dict, b: dict) -> dict:
    """
    The two halves of a line the gutter split, one line again. Where the
    halves differ in kind -- a bold lead word and its sentence -- the line
    is the prose half's kind, so it is read as prose.
    """
    out = dict(b if is_verse_line(a) and not is_verse_line(b) else a)
    out["bbox"] = [min(a["bbox"][0], b["bbox"][0]), min(a["bbox"][1], b["bbox"][1]),
                   max(a["bbox"][2], b["bbox"][2]), max(a["bbox"][3], b["bbox"][3])]
    out["text"] = (a.get("text", "").strip() + " " + b.get("text", "").strip()).strip()
    if a.get("matches") or b.get("matches"):
        out["matches"] = list(a.get("matches") or []) + list(b.get("matches") or [])
    out["words"] = list(a.get("words") or []) + list(b.get("words") or [])
    out["joined"] = True
    out.pop("col", None)
    return out


def bands(lines: list[dict], columns: list, h: float, force: bool = False) -> list[dict]:
    """
    The page's bands, top to bottom, each {"mode", "lines" | "cols"}:

      full     lines across the page (rows rejoined), read in order
      paired   {"verse": [...], "prose": [...]}: a verse column beside its explanation
      columns  {"cols": [[...], [...]]}: two columns of the same sort of text, left then right

    A run of paired rows (with the single lines among them) is a paired band
    when at least MIN_PAIRED_ROWS rows pair and VERSE_SHARE of its left lines
    are verse and of its right lines prose (or the mirror); `force` takes
    every two-column run as paired, verse on the left.
    """
    gutter = columns[0][1]
    # a line across both columns is a full row on its own (the merge splits
    # a line only where the rule runs, so the prose above the band comes
    # whole); the rest pair by height within their columns
    wide = [ln for ln in lines if is_wide(ln, gutter, h)]
    rest = [ln for ln in lines if not is_wide(ln, gutter, h)]
    col0 = [ln for ln in rest if column_of(ln, gutter) == 0]
    col1 = [ln for ln in rest if column_of(ln, gutter) == 1]
    rows = rows_of(col0, col1, h) + [{"left": ln, "right": None, "kind": "full"} for ln in wide]
    rows.sort(key=lambda r: min(x["bbox"][1] for x in (r["left"], r["right"]) if x))
    # runs: a full row ends a two-column run; paired rows and singles extend one
    runs: list[dict] = []
    for r in rows:
        if r["kind"] == "full":
            runs.append({"mode": "full", "rows": [r]})
        elif runs and runs[-1]["mode"] == "two":
            runs[-1]["rows"].append(r)
        else:
            runs.append({"mode": "two", "rows": [r]})
    out: list[dict] = []
    for run in runs:
        rs = run["rows"]
        paired_rows = sum(1 for r in rs if r["kind"] == "paired")
        if run["mode"] == "full" or (paired_rows < MIN_PAIRED_ROWS and not force):
            # a two-column run with fewer than two paired rows is full-width
            # text with a lead word, or a lone line in one column
            flat: list[dict] = []
            for r in rs:
                if r["left"] and r["right"]:
                    flat.append(join_row(r["left"], r["right"]))
                else:
                    flat.append(r["left"] or r["right"])
            if out and out[-1]["mode"] == "full":
                out[-1]["lines"].extend(flat)
            else:
                out.append({"mode": "full", "lines": flat})
            continue
        left = [r["left"] for r in rs if r["left"]]
        right = [r["right"] for r in rs if r["right"]]
        l_verse = sum(1 for ln in left if is_verse_line(ln)) / max(1, len(left))
        r_prose = sum(1 for ln in right if not is_verse_line(ln)) / max(1, len(right))
        r_verse = sum(1 for ln in right if is_verse_line(ln)) / max(1, len(right))
        l_prose = sum(1 for ln in left if not is_verse_line(ln)) / max(1, len(left))
        if force or (l_verse >= VERSE_SHARE and r_prose >= VERSE_SHARE):
            out.append({"mode": "paired", "verse": left, "prose": right})
        elif l_prose >= VERSE_SHARE and r_verse >= VERSE_SHARE:
            out.append({"mode": "paired", "verse": right, "prose": left})
        else:
            out.append({"mode": "columns", "cols": [left, right]})
    return out


VERSE_END = re.compile("॥\\s*[੦-੯0-9]*\\s*॥?\\s*(ਰਹਾਉ\\s*॥?\\s*)?$")
CHUNK_TOL = 0.6      # of the line height: a prose line starting this close to a verse chunk's top starts an explanation
STEP_BREAK = 1.3     # a baseline step this much over the column's usual one starts a new explanation


def verse_chunks(verse: list[dict]) -> list[list[dict]]:
    """The verse column cut after every line that ends a verse (a danda, with or without its number)."""
    chunks: list[list[dict]] = [[]]
    for ln in sorted(verse, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        chunks[-1].append(ln)
        if VERSE_END.search(ln.get("text", "")):
            chunks.append([])
    return [c for c in chunks if c]


def prose_groups(prose: list[dict], verse: list[dict], h: float) -> list[list[dict]]:
    """
    The prose column cut where an explanation begins: level with the top of a
    verse chunk (the arth of a tuk starts beside the tuk), or after a baseline
    step wider than the column's usual one. The paragraph rules then run
    inside each group. Without this the arth of a page whose leading never
    varies is one paragraph, and every verse of the page would be its.
    """
    lines = sorted(prose, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    if len(lines) < 2:
        return [lines] if lines else []
    starts = [c[0]["bbox"][1] for c in verse_chunks(verse)[1:]]
    steps = [b["bbox"][1] - a["bbox"][1] for a, b in zip(lines, lines[1:]) if b["bbox"][1] > a["bbox"][1]]
    usual = statistics.median(steps) if steps else h
    groups: list[list[dict]] = [[lines[0]]]
    for prev, ln in zip(lines, lines[1:]):
        level = any(abs(ln["bbox"][1] - y) <= CHUNK_TOL * h for y in starts)
        wide = (ln["bbox"][1] - prev["bbox"][1]) > STEP_BREAK * usual
        if level or wide:
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


def assign_verse(verse: list[dict], prose_paras: list[dict], h: float) -> list[list[dict]]:
    """
    For each prose paragraph (with "lines"), the verse lines printed beside
    it: the paragraph a verse line overlaps most takes it; a line overlapping
    none goes to the nearest paragraph below it within NEAR line heights
    (the arth starts level with its verse, or a little under), else to the
    paragraph above. Returns one list per paragraph, in the paragraphs'
    order, verse lines in reading order.
    """
    spans = []
    for p in prose_paras:
        ys = [ln["bbox"] for ln in p.get("lines", [])]
        spans.append((min(b[1] for b in ys), max(b[3] for b in ys)) if ys else (0, 0))
    out: list[list[dict]] = [[] for _ in prose_paras]
    for ln in sorted(verse, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        y0, y1 = ln["bbox"][1], ln["bbox"][3]
        best, best_v = -1, 0.0
        for i, (a, b) in enumerate(spans):
            v = max(0, min(y1, b) - max(y0, a)) / max(1.0, float(y1 - y0))
            if v > best_v:
                best, best_v = i, v
        if best < 0:
            below = [(a - y1, i) for i, (a, b) in enumerate(spans) if a >= y1 and a - y1 <= NEAR * h]
            above = [(y0 - b, i) for i, (a, b) in enumerate(spans) if b <= y0]
            if below:
                best = min(below)[1]
            elif above:
                best = min(above)[1]
        if best >= 0:
            out[best].append(ln)
    return out
