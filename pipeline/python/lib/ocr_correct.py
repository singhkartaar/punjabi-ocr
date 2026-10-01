"""
Deterministic correction of a word the lexicon does not know.

An engine reads ਭਾਰੀ as ਬਾਰੀ because ਬ and ਭ differ by one stroke. The lexicon
knows ਭਾਰੀ and, in this case, not ਬਾਰੀ; the candidate one substitution away
along a KNOWN confusion is accepted. That is the whole method (Lehal & Singh
2002 gained three points on clean Gurmukhi print with it), and its discipline
is the same one 12_ingest_writings.py records for the English essays: a wrong
"fix" reads as the author's word and is invisible, while an unknown word is
visibly a scan defect. So a correction is made only when

  - the word is unknown to the lexicon (a known word is never touched),
  - the best candidate IS known,
  - it costs at most MAX_COST in confusion-weighted grapheme edits, and
  - it beats the runner-up by at least MIN_MARGIN.

Two candidates that are both plausible mean the line goes to the arbiter
instead. Nothing here runs on a Gurbani span: those are the corpus's text.

Confusion weights start from SEED_CONFUSIONS -- the pairs an OCR engine
mistakes for one another on printed Gurmukhi (Lehal's sixteen look-alike
subsets and the matra pairs) -- and are then LEARNT from the ground truth:
learn_confusions() aligns each engine's reading of a verified line with the
truth and counts the substitutions, so that what the engines actually get
wrong is what the corrector expects. The table lives in data/ocr/confusions.json.

Words are compared as conjuncts() (lib/ocr_text), not graphemes: a
halant joins its cluster to the next, so a misread subjoined ra is one
cluster's marks away from the word. Split, as graphemes() splits it, the
repair of ਪੁੇਮ to ਪ੍ਰੇਮ cost 2.25 and was never made.
"""
from __future__ import annotations
import json
import os
from collections import Counter

from lib.ocr_text import conjuncts, normalise

MAX_COST = 1.0            # confusion-weighted edits a correction may cost
MIN_MARGIN = 0.5          # by which the best candidate must beat the next
SEED_WEIGHT = 0.5         # cost of a seeded confusion substitution (a plain one costs 1.0)
FREQ_MIN = 5              # corpus occurrences a tie-break winner needs
FREQ_RATIO = 10           # and how many times the runner-up's
AGREED_COST = SEED_WEIGHT  # a word every engine agreed on: one seeded confusion away at most
MARK_COST = 0.75          # one mark changed, added or dropped on the same consonant, unseeded
STACKED = 0.25            # an aunkar stacked on a vowel sign, read for a subjoined ra
NUKTA = "\u0a3c"
AUNKAR = "\u0a41"
DULAINKAR = "\u0a42"
SUBJOIN_MISREADS = (AUNKAR, DULAINKAR)    # what a grey scan makes of the subjoined ra's hook
SUBJOIN_RA = "\u0a4d\u0a30"
VOWEL_SIGNS = set("\u0a3e\u0a3f\u0a40\u0a41\u0a42\u0a47\u0a48\u0a4b\u0a4c")
LEARNT_MIN = 2            # times a substitution must be seen in GT before it is trusted
_STRIP = ".,;:!?\"'()[]{}।॥|-–—"

# Look-alike consonants (Lehal's subsets, the pairs the sample scans showed) and
# the marks engines drop, add or swap. Both directions are meant.
SEED_CONFUSIONS = [
    ("ਬ", "ਭ"),  # ਬ ਭ
    ("ਮ", "ਸ"),  # ਮ ਸ
    ("ਥ", "ਬ"),  # ਥ ਬ
    ("ਖ", "ਘ"),  # ਖ ਘ
    ("ਖ", "ਪ"),  # ਖ ਪ
    ("ਜ", "ਚ"),  # ਜ ਚ
    ("ਜ", "ਲ"),  # ਜ ਲ
    ("ਡ", "ਚ"),  # ਡ ਚ
    ("ਦ", "ਰ"),  # ਦ ਰ
    ("ਨ", "ਲ"),  # ਨ ਲ
    ("ਗ", "ਮ"),  # ਗ ਮ
    ("ਵ", "ਚ"),  # ਵ ਚ
    ("ਤ", "ਨ"),  # ਤ ਨ
    ("ਂ", "ੰ"),  # ਂ ੰ  bindi / tippi
    ("ਿ", "ੀ"),  # ਿ ੀ
    ("ੁ", "ੂ"),  # ੁ ੂ
    ("ੇ", "ੈ"),  # ੇ ੈ
    ("ੋ", "ੌ"),  # ੋ ੌ
    ("ਾ", ""),        # dropped kanna
    ("ਿ", ""),        # dropped sihari
    ("ੁ", ""),        # dropped aunkar
    ("ੰ", ""),        # dropped tippi
    ("ੱ", ""),        # dropped addak
    ("਼", ""),        # dropped nukta
    # The subjoined ra on a grey scan: read as an aunkar (ਪ੍ਰਸ਼ਾਦ as ਪੁਸ਼ਾਦ,
    # ਸ੍ਰੀ as ਸੁੀ) or dropped (ਸੀ, ਗੰਥ). Every ਸ੍ਰੀ, ਗ੍ਰੰਥ, ਪ੍ਰਸ਼ਾਦ and ਪ੍ਰਕਾਸ਼ in the
    # Sant Attar Singh sample (120 dpi) was wrong; the Santhya's 1-bit scan
    # had 5,830 right. Priced as marks after a shared consonant (conjuncts()).
    ("ੁ", "੍ਰ"),      # ੁ ੍ਰ
    ("ੂ", "੍ਰ"),      # ੂ ੍ਰ  (ਕੜਾਹ-ਪੂਸ਼ਾਦ, ਹਿਮਾਚਲ ਪੂਦੇਸ਼)
    ("", "੍ਰ"),       # dropped subjoined ra
]


def _pair_key(a: str, b: str) -> str:
    return "%s>%s" % (a, b)


def _mark_tokens(marks: str) -> list[str]:
    """A cluster's marks as items; a halant and the consonant after it are one."""
    out: list[str] = []
    i = 0
    while i < len(marks):
        if marks[i] == "\u0a4d" and i + 1 < len(marks):
            out.append(marks[i:i + 2])
            i += 2
        else:
            out.append(marks[i])
            i += 1
    return out


def subjoin_variants(word: str) -> list[str]:
    """
    The word with one aunkar (or dulainkar) at a time read back as a
    subjoined ra: the spellings a grey scan's ਪੁਸਾਦਿ, ਬੁਹਮ and ਪੂਦੇਸ਼ were
    printed as (ਪ੍ਰਸਾਦਿ, ਬ੍ਰਹਮ, ਪ੍ਰਦੇਸ਼).
    Both Tesseract variants read those two the same wrong way on the Sant
    Attar Singh sample, so agreement alone would have made them book words.
    """
    return [word[:i] + SUBJOIN_RA + word[i + 1:] for i, ch in enumerate(word) if ch in SUBJOIN_MISREADS]


def subjoin_repairs(word: str) -> list[str]:
    """
    Every spelling one subjoined ra away from the word as read: an aunkar
    or dulainkar read back as the subjoin (subjoin_variants), or the subjoin
    put in after a consonant where none was read (ਗੰਥ -> ਗ੍ਰੰਥ).
    """
    out = subjoin_variants(word)
    for i, ch in enumerate(word):
        if "\u0a15" <= ch <= "\u0a39" and not word[i + 1:i + 2] == "\u0a4d":
            out.append(word[:i + 1] + SUBJOIN_RA + word[i + 1:])
    return out


def shared_misread(word: str, knows, seen: set) -> bool:
    """
    Is a word both engines agreed on a misread subjoined ra? Yes when an
    aunkar or dulainkar in it read back as the subjoin makes a known word
    (ਪੁਸਾਦਿ, ਬੁਹਮ); or when the subjoin put in after a consonant makes a known
    word AND the book also shows that consonant with an aunkar or dulainkar
    (ਗੰਥ, because ਗੁੰਥ is in the readings too). Without that second condition
    ਪਿਆ was refused over Sant Attar Singh vol. 1 for the Kosh's ਪ੍ਰਿਆ: the
    lexicon has little modern Punjabi, so a known ੍ਰ-spelling alone proves
    nothing. `knows` is the lexicon's test, `seen` every word the engines read.
    """
    if knows(word):
        return False
    if any(knows(v) for v in subjoin_variants(word)):
        return True
    for i, ch in enumerate(word):
        if "\u0a15" <= ch <= "\u0a39" and word[i + 1:i + 2] != "\u0a4d":
            if knows(word[:i + 1] + SUBJOIN_RA + word[i + 1:]) and \
                    any(word[:i + 1] + mark + word[i + 1:] in seen for mark in SUBJOIN_MISREADS):
                return True
    return False


def is_subjoin_repair(read: str, best: str) -> bool:
    """
    Does best differ from read only by subjoined ra put back (for an aunkar,
    or where none was read)? ਪੁਸ਼ਾਦ -> ਪ੍ਰਸ਼ਾਦ and ਗੰਥ -> ਗ੍ਰੰਥ are; ਚੱਲਣ -> ਚਲਣ
    and ਖਿਸਕਣ -> ਖਿਸਕਣਾ are not. The one repair a word every engine agreed on
    may receive: measured on the Sant Attar Singh ground truth, the agreed
    corrections of any other kind were 10 of 11 wrong, the dictionary's
    spelling put over the author's.
    """
    r, b = fold(read), fold(best)
    if b in subjoin_variants(r):                      # one aunkar read for the subjoin
        return True
    i = b.find(SUBJOIN_RA)
    while i >= 0:                                     # or the subjoin not read at all
        if b[:i] + b[i + len(SUBJOIN_RA):] == r:
            return True
        i = b.find(SUBJOIN_RA, i + 1)
    return False


def fold(text: str) -> str:
    """Without the nukta: the Kosh and older books write ਪ੍ਰਸਾਦ where a modern one writes ਪ੍ਰਸ਼ਾਦ."""
    return text.replace(NUKTA, "")


def keep_nukta(read: str, best: str) -> str:
    """
    The correction, with the nukta the scan read put back where the
    candidate (folded, or written without it) lacks it at the same cluster:
    ਪੁਸ਼ਾਦ corrected against the Kosh's ਪ੍ਰਸਾਦ is ਪ੍ਰਸ਼ਾਦ, the author's spelling.
    """
    r, b = conjuncts(read), conjuncts(best)
    if len(r) != len(b):
        return best
    out = []
    for x, y in zip(r, b):
        if NUKTA in x and NUKTA not in y and fold(x)[:1] == y[:1]:
            y = y[0] + NUKTA + y[1:]
        out.append(y)
    return "".join(out)


class Confusions:
    """Substitution costs between graphemes; symmetric; 1.0 where unknown."""

    def __init__(self, table: dict | None = None):
        self.cost: dict[str, float] = {}
        for a, b in SEED_CONFUSIONS:
            self.cost[_pair_key(a, b)] = SEED_WEIGHT
            self.cost[_pair_key(b, a)] = SEED_WEIGHT
        for k, v in (table or {}).items():
            self.cost[k] = float(v)

    @classmethod
    def load(cls, path: str) -> "Confusions":
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return cls(json.load(fh).get("cost", {}))
        return cls()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"cost": self.cost}, fh, ensure_ascii=False, indent=1, sort_keys=True)

    def sub(self, a: str, b: str) -> float:
        """Cost of reading a where b was printed, over grapheme clusters."""
        if a == b:
            return 0.0
        c = self.cost.get(_pair_key(a, b))
        if c is not None:
            return c
        # clusters on the same consonant: the cost of turning one's marks into
        # the other's, mark by mark (a single differing mark costs what it did
        # before this was an edit distance: its confusion, else MARK_COST)
        if a and b and a[0] == b[0]:
            return self.marks(a[1:], b[1:])
        return 1.0

    def _mark(self, x: str, y: str) -> float:
        c = self.cost.get(_pair_key(x, y))
        return MARK_COST if c is None else c

    def marks(self, ma: str, mb: str) -> float:
        """
        Edit distance between two clusters' marks, capped at a whole
        substitution (1.0). A subjoined letter (halant + consonant) is one
        mark. Before this, any difference cost a flat MARK_COST, so ਗੁੰ was as
        near ਗਾ (two marks changed) as ਗ੍ਰੰ (one), and the repair of ਗੁੰਥ was
        ambiguous. An aunkar stacked on another vowel sign (ਸੁੀ, ਪੁੇ) is not
        Punjabi spelling; it is a subjoined ra misread, and costs STACKED.
        """
        c = self.cost.get(_pair_key(ma, mb))
        if c is not None:
            return c
        xs, ys = _mark_tokens(ma), _mark_tokens(mb)
        prev = [0.0]
        for y in ys:
            prev.append(prev[-1] + self._mark("", y))
        for i, x in enumerate(xs, start=1):
            stacked = x in SUBJOIN_MISREADS and i < len(xs) and xs[i] in VOWEL_SIGNS
            # a stacked aunkar is a misread shape, not a stray mark: dropping
            # it is no seeded confusion (ਸੁੀ is ਸ੍ਰੀ, not ਸੀ)
            drop = MARK_COST if stacked else self._mark(x, "")
            cur = [prev[0] + drop]
            for j, y in enumerate(ys, start=1):
                sub = 0.0 if x == y else (STACKED if stacked and y == SUBJOIN_RA else self._mark(x, y))
                cur.append(min(prev[j] + drop, cur[j - 1] + self._mark("", y), prev[j - 1] + sub))
            prev = cur
        return min(1.0, prev[-1])


CLUSTER_GAP = 1.5         # dropping or adding a whole consonant cluster is never one "confusion"


def weighted_distance(a: list[str], b: list[str], conf: Confusions, limit: float = 3.0) -> float:
    """
    Edit distance over grapheme lists with confusion-priced substitutions.
    Inserting or deleting a whole cluster costs CLUSTER_GAP: ਪੰਜਵੇਂ is not a
    misreading of ਪੰਜ, whatever the lexicon knows.
    """
    prev = [i * CLUSTER_GAP for i in range(len(b) + 1)]
    for i, x in enumerate(a, start=1):
        cur = [i * CLUSTER_GAP]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + CLUSTER_GAP, cur[j - 1] + CLUSTER_GAP, prev[j - 1] + conf.sub(x, y)))
        if min(cur) > limit:
            return limit + 1.0
        prev = cur
    return prev[-1]


def learn_confusions(pairs: list[tuple[str, str]], base: Confusions | None = None,
                     min_seen: int = LEARNT_MIN) -> Confusions:
    """
    pairs: (engine reading, truth) for verified lines. Substitution counts
    become costs: a pair seen often is cheap, a pair seen once is not trusted.
    """
    from rapidfuzz.distance import Levenshtein
    counts: Counter = Counter()
    for hyp, ref in pairs:
        h, r = conjuncts(normalise(hyp)), conjuncts(normalise(ref))
        for op in Levenshtein.editops(h, r):
            if op.tag == "replace":
                counts[_pair_key(h[op.src_pos], r[op.dest_pos])] += 1
            elif op.tag == "delete":
                counts[_pair_key(h[op.src_pos], "")] += 1
            elif op.tag == "insert":
                counts[_pair_key("", r[op.dest_pos])] += 1
    conf = Confusions(dict(base.cost) if base else None)
    top = max(counts.values(), default=1)
    for key, n in counts.items():
        if n >= min_seen:
            conf.cost[key] = round(max(0.2, 0.6 - 0.4 * n / top), 3)
    return conf


class Corrector:
    def __init__(self, lexicon, confusions: Confusions | None = None):
        self.lex = lexicon
        self.conf = confusions or Confusions()
        self._by_shape: dict | None = None

    def _index(self):
        """Candidates bucketed by first grapheme and length, for a cheap pre-filter."""
        if self._by_shape is None:
            by: dict = {}
            for w in self.lex.candidates():
                g = conjuncts(fold(w))
                if not g:
                    continue
                by.setdefault((g[0][0], len(g)), []).append((w, g))
            self._by_shape = by
        return self._by_shape

    def candidates(self, word: str) -> list[tuple[str, list[str]]]:
        g = conjuncts(word)
        if not g:
            return []
        by = self._index()
        out = []
        firsts = {g[0][0]} | {b[0] for a, b in SEED_CONFUSIONS if a and b and a[0] == g[0][0]} | \
                 {a[0] for a, b in SEED_CONFUSIONS if a and b and b[0] == g[0][0]}
        for f in firsts:
            for n in (len(g) - 1, len(g), len(g) + 1):
                out.extend(by.get((f, n), []))
        return out

    def correct_word(self, word: str, max_cost: float = MAX_COST, allow=None) -> tuple[str, str | None]:
        """
        (word as it should read, why) -- why is None when the word is left
        alone. max_cost and allow(candidate) narrow what may be taken (a word
        every engine agreed on is corrected on stricter terms, correct_line).
        """
        w = normalise(word)
        if not w or self.lex.knows(w):
            return word, None
        fixed, why, cost = self._correct(w, max_cost, allow)
        # the stem repaired, the ending kept: the corrector proposes headwords,
        # and the lexicon knows ਪ੍ਰਸ਼ਾਦਾ only through its suffix rule, so
        # ਪੁਸ਼ਾਦਾ ("a meal") found no candidate at one confusion and 25 stayed
        # in vol. 2 -- or found ਪ੍ਰਸ਼ਾਦ at two, the ending dropped. The cheaper
        # repair wins; on a tie the whole word.
        from lib.ocr_lexicon import _MODERN_SUFFIXES
        best = (cost, fixed) if why == "confusion" else None
        for suf in sorted(_MODERN_SUFFIXES, key=len, reverse=True):
            if w.endswith(suf) and len(conjuncts(w[:-len(suf)])) >= 2:
                ok = (lambda b, suf=suf: allow(b + suf)) if allow is not None else None
                got, stem_why, stem_cost = self._correct(w[:-len(suf)], max_cost, ok)
                if stem_why == "confusion" and self.lex.knows(got + suf) and (best is None or stem_cost < best[0]):
                    best = (stem_cost, got + suf)
        if best is not None:
            return best[1], "confusion"
        return word, why

    def _correct(self, w: str, max_cost: float, allow) -> tuple[str, str | None, float]:
        """correct_word for one normalised form, with no ending taken off."""
        if self.lex.knows(w):
            return w, None, 0.0
        g = conjuncts(fold(w))
        scored = []
        for cand, cg in self.candidates(fold(w)):
            d = weighted_distance(g, cg, self.conf, limit=MAX_COST + 1.0)
            if d <= MAX_COST:
                scored.append((d, cand))
        if not scored:
            return w, None, 0.0
        # one word, two spellings: the Kosh's ਪ੍ਰਸਾਦ and the book's ਪ੍ਰਸ਼ਾਦ
        # are the same candidate, not a tie (keep_nukta restores the book's)
        seen: dict[str, tuple[float, str]] = {}
        for d, cand in sorted(scored):
            seen.setdefault(fold(cand), (d, cand))
        scored = sorted(seen.values())
        best_d, best = scored[0]
        runner = scored[1][0] if len(scored) > 1 else MAX_COST + 1.0
        if runner - best_d < MIN_MARGIN:
            best = self._by_frequency(scored, best_d)
            if best is None:
                return w, "ambiguous", best_d
        if best_d > max_cost or (allow is not None and not allow(best)):
            return w, None, best_d
        return keep_nukta(w, best), "confusion", best_d

    def _by_frequency(self, scored: list[tuple[float, str]], best_d: float) -> str | None:
        """
        Two words equally near the reading: the one the corpus uses FREQ_RATIO
        times as often (and at least FREQ_MIN times), if it is among the
        nearest; else None. ਬੁਹਮ is one confusion from ਬ੍ਰਹਮ and from the
        Kosh's ਬਹਮ, ਪੇਮ from ਪ੍ਰੇਮ and ਪੇਸ; the scan's error and the corpus
        both say which.
        """
        freq = getattr(self.lex, "freq", None)
        if not freq:
            return None
        close = [(freq.get(c, 0), d, c) for d, c in scored if d - best_d < MIN_MARGIN]
        close.sort(key=lambda x: -x[0])
        (f1, d1, c1), f2 = close[0], (close[1][0] if len(close) > 1 else 0)
        if d1 == best_d and f1 >= FREQ_MIN and f1 >= FREQ_RATIO * max(f2, 1):
            return c1
        return None

    def correct_line(self, text: str, protected: list[tuple[int, int]] | None = None,
                     only: set[str] | None = None, agreed: set[str] | None = None,
                     agreed_ok=None) -> tuple[str, list]:
        """
        (text, [[was, now, why], ...]). protected: character spans of the
        normalised text that are corpus Gurbani and must not change. only:
        when given, just these words (as read) are candidates -- the merge
        passes the words the engines disagreed on. agreed: words every engine
        read the same way, corrected only one seeded confusion away
        (AGREED_COST) and only to a word agreed_ok(word) accepts; recorded as
        "confusion-agreed". Punctuation round a word is kept, not compared.
        """
        s = normalise(text)
        out, changes, pos = [], [], 0
        for tok in s.split(" "):
            start, end = pos, pos + len(tok)
            pos = end + 1
            if not tok or any(a < end and start < b for a, b in (protected or [])):
                out.append(tok)
                continue
            core = tok.strip(_STRIP)
            if not core:
                out.append(tok)
                continue
            lead = tok[:tok.index(core)]
            trail = tok[len(lead) + len(core):]
            if only is None or core in only:
                strict = False
            elif agreed is not None and core in agreed:
                strict = True
            else:
                out.append(tok)
                continue
            # a compound (ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ, ਛਕਦਿਆਂ-ਛਕਦਿਆਂ) is corrected part by part
            parts = []
            for part in core.split("-"):
                if strict:
                    ok = (lambda b, part=part: is_subjoin_repair(part, b) and (agreed_ok is None or agreed_ok(b)))
                    fixed, why = self.correct_word(part, max_cost=AGREED_COST, allow=ok)
                    why = "confusion-agreed" if why == "confusion" else why
                else:
                    fixed, why = self.correct_word(part)
                if part and why in ("confusion", "confusion-agreed") and fixed != part:
                    changes.append([part, fixed, why])
                    parts.append(fixed)
                else:
                    parts.append(part)
            out.append(lead + "-".join(parts) + trail)
        return " ".join(out), changes
