"""
A commentary's text, line by line: what a teeka view shows beside each line.

  30_line_commentary.py --src data/santhya --work santhya --translator pa-santhya
  30_line_commentary.py ... --manifest data/books/santhya/manifest.json   # each volume's own angs only
  30_line_commentary.py --src data/santhya --work santhya --print 5

A commentary OCR'd from its printed pages (kind `commentary`, a paired
layout) arrives from 12_ingest_writings.py with each paragraph's `explains`:
the line or lines of the Granth it was printed beside. The Santhya prints a
line and then Bhai Vir Singh's explanation of it, line after line, so 17,874
of its 19,171 links name one line. That is the shape of a teeka -- one text
per line, like the Faridkot Teeka -- and this step writes it so:

  <src>/lines-<translator>.jsonl    {"line_id", "text", "pages": ["2:143", ...]}

which pipeline/node/src/06-build-translations-db.js imports into
translations.sqlite under that translator id, for server.js to offer as a
view (?tr=vs).

Every body paragraph that explains a line is joined to that line's text in
reading order; one that explains two or more lines goes on the last of them,
where the explanation ends. What is not prose is left out: a paragraph with
fewer than two Gurmukhi words, or mostly digits and marks (the page's stray
numerals and smudges the OCR read, "2੭", "4."), and footnotes, which belong
to a word, not the line. The text is the OCR's, at the volume's measured
accuracy; the report says so and the view says where it came from.

WHERE ONE LINE'S EXPLANATION ENDS. The page is a stream: the explanation runs
on under the lines it explains, and the OCR's paragraphs break where the
printed line wraps, not where a line's explanation ends -- so a paragraph
often opens with the tail of the line before ("(ਲਗ ਰਹੀ ਹੈ)। (ਉਸ ਸਭਾ ਵਿਚ)
...") and stops mid-sentence ("... ਉਸੇ ਨੂੰ ਆਦਰ ਮਿਲੇਗਾ ਜਿਸ"). And where a
stray verse number ("੨॥") was read as a line of the Granth, the paragraph
after it was left unpaired and its words lost ("ਉਸ ਸੁਖ ਸਾਗਰ ਦੀ" without its
"ਸੇਵਾ ਕਰ।"). Bhai Vir Singh closes each line's explanation with a danda, and
a verse's with its number, "॥੧॥", as the line itself closes. So a shabad's
explanation is read as ONE stream, the unpaired paragraphs in it, and cut
again (`recut`), once between each two lines, near where the OCR broke it:
best right after a "।" or "॥", better still after the verse number the line
ends with, never inside a bracket, and otherwise where the words either side
echo their own line's words ("ਆਠੇ ਪਹਿਰ ... ਗੋਬਿੰਦ" for "ਆਠ ਪਹਰ ਗੋਵਿੰਦੁ").

FROM THE RAHAO. Some shabads he explains from the rahao, and says so: "(ਅਰਥ
ਰਹਾਉ ਤੋਂ ਟੁਰੇਗਾ)". The OCR still paired his paragraphs with the lines in the
order they are printed, so the rahao's explanation sat beside the first
verse. There the stream is cut in the order he explains (`rahao_order`):
the rahao, then the first verse, whose lines may stay empty where the
stream does not reach them; the note itself is dropped.
"""
from __future__ import annotations
import argparse
import bisect
import json
import os
import re
import sqlite3
from collections import defaultdict


GURMUKHI = re.compile(r"[਀-੿]")
LETTER = re.compile(r"[\u0A05-\u0A39\u0A59-\u0A5E]")             # a letter, not a vowel sign or a mark
MIN_WORDS = 2
MIN_GURMUKHI_SHARE = 0.5

# recut: what a cut is worth, against one word echoing its own line (1.0)
CUT_VERSE = 6.0          # after "॥੧॥", the number the line ends with
CUT_OTHER_VERSE = 0.0    # after a verse number the line does NOT end with: another line's end
CUT_STOP = 3.0           # after a "।" or "॥" or "?"
CUT_PAUSE = 0.3          # after a comma or a semicolon (a list's commas too: little)
CUT_GLOSS = 0.8          # before "(": a line's explanation so often opens with one, "(ਤਾਂ ਤੇ)"
CUT_IN_BRACKET = -4.0    # inside "( ... )": the gloss is one thought
BRACKET_MAX = 30         # words a bracket may stay open before it is taken as never closed
SHORT = -0.5             # each word a line's explanation falls short of the line's own count
REACH = 2                # paragraphs either side of its own a cut may move into
CUT_AS_PRINTED = 0.3     # where the OCR broke it, all else equal
MAX_SKIPPED = 3          # lines without a paragraph that one stream may pass over
STOP_END = re.compile(r"[।॥?]['\"’”)\]]*$")
PAUSE_END = re.compile(r"[,;:]['\"’”)\]]*$")
VERSE_NO = re.compile(r"(?:॥|^)\s*([੦-੯]+)\s*॥")      # "॥੩॥", or "੩॥" with its first mark lost
# two-consonant words of the prose that echo nothing ("ਵਿਚ", "ਨਾਲ", "ਕਰ")
# "(ਅਰਥ ਰਹਾਉ ਤੋਂ ਟੁਰੇਗਾ)", "(ਰਹਾਉ ਤੋਂ ਅਰਥ ਕਰਨਾ ਹੈ)", "(ਰਹਾਉ ਦਾ ਅਰਥ ਪਹਿਲੋਂ ਲੱਗੇਗਾ)": the
# shabad's meaning is given from its rahao, then the first verse
RAHAO_FIRST = re.compile(r"\((?=[^()]*ਰਥ)[^()]{0,25}ਰਹਾਉਂ?\s+(?:ਦੀ ਦੂਸਰੀ ਤੁਕ\s+)?(?:ਤੋਂ|ਦਾ ਅਰਥ)[^()]{0,25}\)")
PROSE_STOP ={"ਹਨ", "ਵਚ", "ਨਲ", "ਜਸ", "ਤਸ", "ਅਸ", "ਅਹ", "ਕਰ", "ਜਦ", "ਤਦ", "ਹਣ", "ਭਵ", "ਅਤ", "ਕਹ", "ਜਣ"}
# the nukta letters, precomposed, to their plain consonant (a combining nukta is dropped anyway)
NUKTA = str.maketrans({"ਖ਼": "ਖ", "ਗ਼": "ਗ", "ਜ਼": "ਜ", "ਫ਼": "ਫ",
                       "ਸ਼": "ਸ", "ਲ਼": "ਲ"})


def is_word(token: str) -> bool:
    """Two letters or more, with whatever vowel signs sit between them: a word, not a mark."""
    return len(LETTER.findall(token)) >= 2


def is_prose(text: str) -> bool:
    """A paragraph of Punjabi prose, not a stray numeral or smudge the OCR read."""
    ink = [c for c in text if not c.isspace()]
    if not ink:
        return False
    share = sum(1 for c in ink if GURMUKHI.match(c)) / len(ink)
    words = sum(1 for w in text.split() if is_word(w))
    return words >= MIN_WORDS and share >= MIN_GURMUKHI_SHARE


def is_continuation(text: str) -> bool:
    """An unpaired paragraph worth keeping in the stream: a word, or a verse number."""
    return any(is_word(w) for w in text.split()) or bool(VERSE_NO.search(text))


def skeleton(word: str) -> str:
    """A word's consonants, nukta folded and an opening vowel as ਅ: ਗੋਵਿੰਦੁ -> ਗਵਦ, ਆਠੇ -> ਅਠ."""
    out = []
    for i, c in enumerate(word.translate(NUKTA)):
        if "ਕ" <= c <= "ਹ" or c == "ੜ":
            out.append(c)
        elif i == 0 and ("ਅ" <= c <= "ਔ" or c in "ੲੳ"):
            out.append("ਅ")
    return "".join(out)


def one_apart(a: str, b: str) -> bool:
    """Equal, or one letter changed, added or dropped."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def echoes(sk: str, line: set[str]) -> bool:
    """Whether a word of the prose (as its skeleton) renders a word of the line."""
    if len(sk) < 2 or sk in PROSE_STOP:
        return False
    if sk in line:
        return True
    # one letter apart (ਗੋਬਿੰਦ, ਗੋਵਿੰਦੁ), or one stem (ਬਖਸ਼ਣਹਾਰ, ਬਖਸਿੰਦੁ)
    return len(sk) >= 3 and any(len(g) >= 3 and g[0] == sk[0] and (
        one_apart(sk, g) or (min(len(sk), len(g)) >= 4 and sk[:3] == g[:3])) for g in line)


def verse_number(text: str) -> str | None:
    m = VERSE_NO.search(text)
    return m.group(1) if m else None


class Window:
    """Running counts from word `start` on: w[c] is the count over words start..c-1."""

    def __init__(self, start: int):
        self.start, self.counts = start, [0]

    def __getitem__(self, c: int) -> int:
        return self.counts[c - self.start]


def recut(words: list[str], starts: list[int], slots: list[str],
          optional: list[bool] | None = None, reach: int = REACH,
          loose: list[bool] | None = None) -> list[int]:
    """
    Where each line's explanation starts in one shabad's stream of words.

    `starts` are where the OCR's paragraphs put them (starts[0] == 0), `slots`
    the Granth text of each line (or lines) explained. Each cut moves at most
    `reach` paragraphs either side of it and every line keeps a word -- but
    an `optional` one, a line no paragraph was printed beside (its start is
    the next line's), which may stay empty, and a `loose` one even at the
    stream's either end; the cuts maximise the words that
    echo their own line, plus what each cut is worth where it falls (CUT_*).
    """
    n, m = len(slots), len(words)
    if n < 2:
        return starts[:]
    sks = [skeleton(w.strip("()[]{}'\"‘’“”,.;:!?।॥-")) for w in words]
    lines = [{skeleton(w) for w in s.split()} - {""} for s in slots]
    nos = [verse_number(s) for s in slots]
    # depth[c]: brackets open before word c. A gloss never runs past a "।" or
    # "॥", nor far: a ")" the OCR lost ("(ਕਰਮਾਂ ਕਰਕੇ ਕੂੜਿਆਰ= । ਝੁਠ ...") must
    # not make the rest of the stream one long bracket
    depth, d, opened = [0] * (m + 1), 0, 0
    for i, w in enumerate(words):
        d = max(0, d + w.count("(") - w.count(")"))
        opened = opened + 1 if d else 0
        if STOP_END.search(w) or opened > BRACKET_MAX:
            d, opened = 0, 0
        depth[i + 1] = d
    def worth(c: int, k: int) -> float:
        """A cut before word c, closing slot k."""
        w = words[c - 1]
        v = CUT_AS_PRINTED if c == starts[k + 1] else 0.0
        if depth[c] > 0:
            v += CUT_IN_BRACKET
        if STOP_END.search(w):
            no = verse_number(w)
            v += CUT_STOP if not no else CUT_VERSE if no == nos[k] else CUT_OTHER_VERSE
        elif PAUSE_END.search(w):
            v += CUT_PAUSE
        if c < m and depth[c] == 0 and words[c].startswith("("):
            v += CUT_GLOSS
        return v

    size = [sum(1 for w in s.split() if is_word(w)) for s in slots]

    def short(length: int, k: int) -> float:
        """An explanation shorter than the line it explains is a cut in the wrong place."""
        return SHORT * max(0, size[k] - length)

    optional = optional or [False] * n
    loose = loose or [False] * n
    # where cut k (the start of slot k) may fall: from just past the paragraph
    # start before its own to just short of the one after
    marks = sorted(set(starts)) + [m]
    rng = [(0, 0)]
    for k in range(1, n):
        i = bisect.bisect_left(marks, starts[k])          # marks[i] == starts[k]
        # one word past the mark and one short of the next, so the slots either side keep
        # one -- unless it may stay empty
        rng.append((marks[max(0, i - reach)] + (0 if loose[k - 1] else 1),
                    marks[min(len(marks) - 1, i + reach)] - (0 if loose[k] else 1)))
    rng.append((m, m))
    # pre[k][c]: words before c that echo slot k -- counted only over the
    # stretch slot k can reach, from its earliest start to its latest end
    pre = []
    for k in range(n):
        a, b = rng[k][0], rng[k + 1][1]
        p, run = Window(a), 0
        for i in range(a, b):
            run += 1 if echoes(sks[i], lines[k]) else 0
            p.counts.append(run)
        pre.append(p)
    best = {0: 0.0}
    back: list[dict] = [{}]
    for k in range(1, n):
        lo, hi = rng[k]
        p, near = pre[k - 1], max(1, size[k - 1])
        prev = sorted(best)
        cur, bk = {}, {}
        j, run_top, run_arg = 0, None, None               # best s0 - p[c0] over c0 <= c - near
        for c in range(lo, hi + 1):
            while j < len(prev) and prev[j] <= c - near:
                v = best[prev[j]] - p[prev[j]]
                if run_top is None or v > run_top:
                    run_top, run_arg = v, prev[j]
                j += 1
            top, arg = None, None
            if run_top is not None:                       # far enough back: no shortfall
                top, arg = run_top + p[c] + worth(c, k - 1), run_arg
            # the few closer starts, short of the line's own length
            for c0 in prev[bisect.bisect_right(prev, c - near):bisect.bisect_right(prev, c)]:
                if c0 == c:
                    if not optional[k - 1]:
                        continue          # every line keeps a word, but one printed without any
                    s = best[c0]
                else:
                    s = best[c0] + p[c] - p[c0] + worth(c, k - 1) + short(c - c0, k - 1)
                if top is None or s > top:
                    top, arg = s, c0
            if top is not None:
                cur[c], bk[c] = top, arg
        if not cur:                       # nothing fits: keep the OCR's cuts
            return starts[:]
        best = cur
        back.append(bk)
    c = max(best, key=lambda x: best[x] + pre[n - 1][m] - pre[n - 1][x]
            + (0.0 if x == m and loose[n - 1] else short(m - x, n - 1)))
    cuts = [c]
    for k in range(n - 1, 1, -1):
        c = back[k][c]
        cuts.append(c)
    return [0] + cuts[::-1]


def rahao_order(slots: list[dict], kinds: dict[int, tuple[str, int]],
                granth: dict[int, str]) -> tuple[list[int], set[int], set[int], int]:
    """
    The order a stream's slots are explained in: as printed, except where
    Bhai Vir Singh says "(ਅਰਥ ਰਹਾਉ ਤੋਂ ਟੁਰੇਗਾ)" -- there the shabad's rahao
    (the lines from the end of the verse before it through the rahao line)
    is explained first, then the first verse. Returns the order, the slots
    whose note was acted on, and the shabad's invocation and heading slots
    before its first verse, which in such a shabad have nothing of their own,
    and the most lines a rahao moved past: the paragraphs printed beside the
    verse hold the rahao's explanation, so a cut may fall that much further
    from where the OCR broke.
    """
    order, marked, heads, jump = list(range(len(slots))), set(), set(), 0
    kind = lambda k: kinds.get(slots[k]["lines"][0], (None, None))
    notes = [i for i, s in enumerate(slots) if RAHAO_FIRST.search(" ".join(t for t, _ in s["parts"]))]
    for i in notes:
        n = len(slots)
        j = next((k for k in range(i, n) if kind(k)[0] in ("line", "rahao")), None)
        if j is None:
            continue
        shabad = kind(j)[1]
        r = next((k for k in range(j, n) if kind(k)[1] != shabad or kind(k)[0] == "rahao"), None)
        if r is None:
            # the stream stopped short of the rahao, its line never paired (read into a
            # paragraph): its explanation is in the stream already, so its slot joins it
            last = slots[-1]["lines"][-1]
            at = next((x for x in range(last + 1, last + 2 + MAX_SKIPPED)
                       if kinds.get(x, (None, None))[1] == shabad and kinds[x][0] == "rahao"), None)
            if at is None or any(kinds.get(x, (None, None))[1] != shabad for x in range(last + 1, at)):
                continue
            for x in range(last + 1, at + 1):
                slots.append({"lines": [x], "parts": [], "optional": True})
                order.append(len(slots) - 1)
            r = len(slots) - 1
        if r is None or kind(r)[1] != shabad:
            continue                      # the rahao is not in this stream
        b = j
        for k in range(j, r):
            if verse_number(granth[slots[k]["lines"][-1]]):
                b = k + 1                 # the rahao block starts after the last verse's end
        if b == j:
            continue                      # the rahao already comes first
        # the verse's explanation follows the rahao's, often cut short where the OCR
        # lost a pairing or a paragraph ends the stream: what the stream has goes to the
        # lines it echoes, and a line it does not reach stays empty rather than take scraps
        heads.update(range(j, b))
        block = list(range(b, r + 1))
        jump = max(jump, b - j)
        order = [x for x in order if x not in block]
        at = order.index(j)
        order[at:at] = block
        marked.add(i)
        heads.update(k for k in range(i, j) if kind(k)[0] in ("invocation", "heading"))
    return order, marked, heads, jump


def line_texts(records: list[dict], angs: dict[int, list[int]] | None = None,
               granth: dict[int, str] | None = None,
               kinds: dict[int, tuple[str, int]] | None = None) -> tuple[dict[int, dict], dict[str, int]]:
    """
    {line_id: {"text", "pages"}} from the paragraph records, in reading order,
    and how many paragraphs were taken or left out, by reason.

    `angs` {part: [from, to]}, when given, keeps a volume to the lines of its own
    angs. Off for the Santhya (the owner's call, 2026-10-01): a link outside
    them is a verse the commentary quotes and explains in passing (vol. 1's
    Japji commentary on ang 929's ਓਅੰਕਾਰਿ ਬੇਦ ਨਿਰਮਏ), 36 paragraphs, and what
    he says there is worth having beside that line too.

    `granth` {line_id: Gurmukhi}, when given, re-cuts each stream between its
    lines (`recut`); without it a line keeps the paragraphs printed beside it,
    an unpaired one going on with the line before. `kinds` {line_id: (kind,
    shabad_id)}, with it, lets a shabad explained from its rahao be cut in
    that order (`rahao_order`).
    """
    lines: dict[int, dict] = defaultdict(lambda: {"parts": [], "pages": []})
    seen = defaultdict(int)
    run: list[dict] = []                  # one stream: [{"lines": [ids], "parts": [(text, page)]}]

    def close():
        if not run:
            return
        slots = run[:]
        run.clear()
        if granth:
            # a line no paragraph was printed beside -- between two explained
            # ones, or the second of two the OCR read as one quote: its
            # explanation is in the stream somewhere near, and the cut finds it
            full = []
            for s in slots:
                if full:
                    full += [{"lines": [i], "parts": [], "optional": True}
                             for i in range(full[-1]["lines"][-1] + 1, s["lines"][0])]
                full.append({**s, "lines": s["lines"][:1]})
                full += [{"lines": [i], "parts": [], "optional": True} for i in s["lines"][1:]]
            slots = full
        if granth and len(slots) > 1 and all(i in granth for s in slots for i in s["lines"]):
            order, marked, heads, jump = rahao_order(slots, kinds, granth) if kinds else (list(range(len(slots))), set(), set(), 0)
            for k in marked:              # the note says only what the order now does
                slots[k]["parts"] = [(RAHAO_FIRST.sub(" ", t), p) for t, p in slots[k]["parts"]]
                seen["explained from the rahao"] += 1
            # the words stay as printed; only which line each stretch explains is reordered
            words, pages, starts = [], [], []
            for s in slots:
                starts.append(len(words))
                for text, page in s["parts"]:
                    for w in text.split():
                        words.append(w)
                        pages.append(page)
            cuts = recut(words, starts, [" ".join(granth[i] for i in slots[o]["lines"]) for o in order],
                         [bool(slots[o].get("optional")) or o in heads for o in order], REACH + jump,
                         [o in heads for o in order]) + [len(words)]
            # the stream's last line ends at its own verse number; what follows
            # it is the next note, run on (" ॥੯॥੧੪॥। ਨੇ ਗੁਰੂ ਜੀ ਨਾਲ ਕਾਲ ਨੂੰ...")
            no = verse_number(granth[slots[order[-1]]["lines"][-1]])
            ends = [i for i in range(cuts[-2], cuts[-1]) if no and verse_number(words[i]) == no]
            if ends and ends[-1] + 1 < cuts[-1]:
                seen["words after a verse's end"] += cuts[-1] - ends[-1] - 1
                cuts[-1] = ends[-1] + 1
            for k, o in enumerate(order):
                s = slots[o]
                seg = range(cuts[k], cuts[k + 1])
                parts = [(" ".join(words[i] for i in seg), None)]
                s["parts"] = parts
                s["pages"] = list(dict.fromkeys(pages[i] for i in seg))
        for s in slots:
            text = " ".join(t for t, _ in s["parts"] if t)
            if not text:
                continue                  # a line the stream had nothing for
            line = s["lines"][-1]
            lines[line]["parts"].append(text)
            for page in s.get("pages") or [p for _, p in s["parts"]]:
                if page not in lines[line]["pages"]:
                    lines[line]["pages"].append(page)

    for rec in records:
        explains = rec.get("explains") or []
        ex = [e for e in explains if e.get("line_to") or e.get("line_from")]
        page = "%s:%s" % (rec.get("part") or 0, rec["page"])
        text = " ".join(rec["text"].split())
        if not ex:
            if explains and rec.get("style") == "body":
                # paired layout, but its Granth line was a stray number: the stream goes on
                if run and is_continuation(text):
                    run[-1]["parts"].append((text, page))
                    seen["continued"] += 1
            elif rec.get("style") not in ("quote", "footnote"):
                close()                   # the explanation's notes, a heading: the stream ends
            continue
        if rec.get("style") not in ("body",):
            seen["not body (%s)" % rec.get("style")] += 1
            continue
        if (ex[0].get("source") or "G") != "G":
            seen["another scripture"] += 1
            close()
            continue
        span = (angs or {}).get(rec.get("part") or 0)
        if span and ex[-1].get("ang") and not (span[0] - 1 <= ex[-1]["ang"] <= span[-1] + 1):
            seen["outside the volume's angs"] += 1
            close()
            continue
        if not is_prose(text):
            seen["not prose"] += 1
            continue
        lo = ex[0].get("line_from") or ex[0]["line_to"]
        hi = ex[-1].get("line_to") or ex[-1]["line_from"]
        if run and run[-1]["lines"][-1] == hi:
            run[-1]["parts"].append((text, page))
        else:
            # back, or on past lines nothing explains: a new stream. Into the next
            # shabad is the same stream -- its verse number says where one ends
            if run and not (run[-1]["lines"][-1] < lo <= run[-1]["lines"][-1] + 1 + MAX_SKIPPED):
                close()
            run.append({"lines": list(range(lo, hi + 1)), "parts": [(text, page)]})
        seen["taken"] += 1
    close()
    out = {line: {"text": " ".join(v["parts"]), "pages": v["pages"]} for line, v in lines.items()}
    return out, dict(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the corpus folder with <work>.jsonl (data/santhya)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--translator", default=None, help="the translator id it is imported as (default pa-<work>)")
    ap.add_argument("--label", default=None, help="what a reader is shown it is (default: the work's title)")
    ap.add_argument("--manifest", help="the books' manifest.json: each part's `angs` bounds the lines it explains")
    ap.add_argument("--as-printed", action="store_true",
                    help="keep the OCR's paragraph breaks between lines instead of re-cutting each stream")
    ap.add_argument("--print", dest="show", type=int, default=0)
    args = ap.parse_args()
    src = os.path.abspath(args.src)
    translator = args.translator or "pa-%s" % args.work
    with open(os.path.join(src, args.work + ".jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    meta, records = rows[0]["_meta"], rows[1:]
    angs = None
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as fh:
            angs = {w.get("part") or 0: w["angs"] for w in json.load(fh)["works"] if w.get("angs")}
    granth = kinds = None
    if not args.as_printed:
        from lib.paths import CORPUS_DB
        con = sqlite3.connect(CORPUS_DB)
        granth = dict(con.execute("SELECT line_id, gurmukhi_uni FROM lines"))
        kinds = {i: (k, s) for i, k, s in con.execute("SELECT line_id, kind, shabad_id FROM lines")}
        con.close()
    lines, seen = line_texts(records, angs, granth, kinds)
    out = os.path.join(src, "lines-%s.jsonl" % translator)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": {"translator": translator, "lang": meta.get("language", "pa"),
                                       "kind": "teeka", "label": args.label or meta.get("title"),
                                       "author": meta.get("author"), "work": args.work,
                                       "origin": "OCR of the printed volumes (%s)" % ", ".join(meta.get("files", [])),
                                       "lines": len(lines), "paragraphs": seen}}, ensure_ascii=False) + "\n")
        for line in sorted(lines):
            fh.write(json.dumps({"line_id": line, **lines[line]}, ensure_ascii=False) + "\n")
    words = sum(len(v["text"].split()) for v in lines.values())
    print("%s: %d lines with commentary, %d words; paragraphs %s -> %s"
          % (args.work, len(lines), words, json.dumps(seen, ensure_ascii=False), out))
    for line in sorted(lines)[:args.show]:
        print("  %d [%s] %s" % (line, ",".join(lines[line]["pages"]), lines[line]["text"][:200]))


if __name__ == "__main__":
    main()
