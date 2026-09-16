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
"""
from __future__ import annotations
import json
import os
from collections import Counter

from lib.ocr_text import graphemes, normalise

MAX_COST = 1.0            # confusion-weighted edits a correction may cost
MIN_MARGIN = 0.5          # by which the best candidate must beat the next
MAX_CANDIDATES = 200
SEED_WEIGHT = 0.5         # cost of a seeded confusion substitution (a plain one costs 1.0)
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
]


def _pair_key(a: str, b: str) -> str:
    return "%s>%s" % (a, b)


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
        # a cluster differing only in a mark: cost of that mark's confusion
        if a and b and a[0] == b[0]:
            ma, mb = a[1:], b[1:]
            c = self.cost.get(_pair_key(ma, mb))
            if c is not None:
                return c
            if not ma or not mb:
                c = self.cost.get(_pair_key(ma or mb, ""))
                if c is not None:
                    return c
            return 0.75
        return 1.0


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
        h, r = graphemes(normalise(hyp)), graphemes(normalise(ref))
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
                g = graphemes(w)
                if not g:
                    continue
                by.setdefault((g[0][0], len(g)), []).append((w, g))
            self._by_shape = by
        return self._by_shape

    def candidates(self, word: str) -> list[tuple[str, list[str]]]:
        g = graphemes(word)
        if not g:
            return []
        by = self._index()
        out = []
        firsts = {g[0][0]} | {b[0] for a, b in SEED_CONFUSIONS if a and b and a[0] == g[0][0]} | \
                 {a[0] for a, b in SEED_CONFUSIONS if a and b and b[0] == g[0][0]}
        for f in firsts:
            for n in (len(g) - 1, len(g), len(g) + 1):
                out.extend(by.get((f, n), []))
        return out[:MAX_CANDIDATES * 5]

    def correct_word(self, word: str) -> tuple[str, str | None]:
        """(word as it should read, why) -- why is None when the word is left alone."""
        w = normalise(word)
        if not w or self.lex.knows(w):
            return word, None
        g = graphemes(w)
        scored = []
        for cand, cg in self.candidates(w):
            d = weighted_distance(g, cg, self.conf, limit=MAX_COST + 1.0)
            if d <= MAX_COST:
                scored.append((d, cand))
        if not scored:
            return word, None
        scored.sort()
        best_d, best = scored[0]
        runner = scored[1][0] if len(scored) > 1 else MAX_COST + 1.0
        if runner - best_d < MIN_MARGIN:
            return word, "ambiguous"
        return best, "confusion"

    def correct_line(self, text: str, protected: list[tuple[int, int]] | None = None,
                     only: set[str] | None = None) -> tuple[str, list]:
        """
        (text, [[was, now, why], ...]). protected: character spans of the
        normalised text that are corpus Gurbani and must not change. only:
        when given, just these words (as read) are candidates -- the merge
        passes the words the engines disagreed on.
        """
        s = normalise(text)
        out, changes, pos = [], [], 0
        for tok in s.split(" "):
            start, end = pos, pos + len(tok)
            pos = end + 1
            if not tok or any(a < end and start < b for a, b in (protected or [])):
                out.append(tok)
                continue
            if only is not None and tok.strip(_STRIP) not in only:
                out.append(tok)
                continue
            fixed, why = self.correct_word(tok)
            if why == "confusion" and fixed != tok:
                changes.append([tok, fixed, why])
                out.append(fixed)
            else:
                out.append(tok)
        return " ".join(out), changes
