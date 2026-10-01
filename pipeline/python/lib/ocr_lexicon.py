"""
Is this a word? The lexicon gate for OCR'd commentary.

A word an engine read is accepted when a source that owes nothing to the scan
knows it. That principle is the one 12_ingest_writings.py records for the
English essays: the scan's own text cannot say whether "imran" is a word,
because its defect is systematic; an outside vocabulary can. Here the outside
vocabularies are:

  Mahan Kosh       56,978 headwords, with lib/mahankosh.py's one-suffix
                   stemmer, so ਨਾਮੁ / ਨਾਮਿ / ਨਾਮੈ resolve to ਨਾਮ
  SGGS             every token of the corpus, so a Gurbani word quoted inside
                   a commentary sentence is not flagged
  Darpan prose     Sahib Singh's own modern Punjabi (translations.pa-ss in
                   corpus.sqlite), the same register as the books, which the
                   1930 Kosh alone would leave with a high false-OOV rate
  English          the words of the three English translations (english_words()
                   in 12_ingest_writings.py), for the English books and the
                   inline English terms in the Punjabi ones
  the book itself  words that at least two engines read the same way at least
                   MIN_SEEN times: the author's own vocabulary, admitted only
                   on independent agreement, never on one engine's repetition

The gate answers one question -- known or not -- and reports the share of a
line's words it does not know. Correction (lib/ocr_correct.py) and routing
(lib/ocr_route.py) act on that; nothing here changes text.
"""
from __future__ import annotations
import os
import re
import sqlite3
from collections import Counter
from importlib import import_module

from lib.ocr_text import GURMUKHI, LATIN, graphemes, normalise, strip_footnote_marker, words

MIN_SEEN = 3
DARPAN_MIN = 3
_PUNCT = "।॥|.,;:!?\"'()[]{}-–—‘’“”"
_NUMERIC = re.compile("^[0-9੦-੯.:/-]+$")
_STACKED = re.compile("[\u0a3e-\u0a42\u0a47\u0a48\u0a4b\u0a4c]{2}")
_NUKTA = "\u0a3c"
# Modern Punjabi inflections a headword list does not carry: plural/oblique
# endings that the Kosh's Gurbani-era stemmer does not know.
_MODERN_SUFFIXES = ["ਾਂ", "ੀਂ", "ੁਂ", "ੇ", "ਾ", "ੀ", "ਿਆਂ", "ਿਆ"]


class Lexicon:
    def __init__(self):
        self.kosh = None
        self.gurbani: set[str] = set()
        self.modern: set[str] = set()
        self.english: set[str] = set()
        self.book: set[str] = set()
        # how often a word occurs in the corpus (Gurbani and the Darpan's
        # Punjabi): the corrector's tie-break between two equally near words
        self.freq: Counter = Counter()
        self.sources: dict[str, int] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_sources(cls, kosh_db: str | None = None, corpus_db: str | None = None,
                     english: bool = True) -> "Lexicon":
        lex = cls()
        if kosh_db and os.path.exists(kosh_db) and os.path.getsize(kosh_db) > 0:
            from lib.mahankosh import MahanKosh
            lex.kosh = MahanKosh(kosh_db)
            lex.sources["kosh"] = lex.kosh.count()
        if corpus_db and os.path.exists(corpus_db) and os.path.getsize(corpus_db) > 0:
            con = sqlite3.connect(corpus_db)
            try:
                from lib.gurmukhi_text import clean_gurmukhi
                for (text,) in con.execute("SELECT gurmukhi_uni FROM lines"):
                    ws = words(clean_gurmukhi(text or ""))
                    lex.gurbani.update(ws)
                    lex.freq.update(ws)
                lex.sources["gurbani"] = len(lex.gurbani)
                counts: Counter = Counter()
                try:
                    for (text,) in con.execute("SELECT text FROM translations WHERE translator='pa-ss'"):
                        counts.update(w for w in words(text or "") if GURMUKHI.search(w))
                except sqlite3.OperationalError:
                    pass
                lex.modern = {w for w, n in counts.items() if n >= DARPAN_MIN}
                lex.freq.update(counts)
                lex.sources["darpan"] = len(lex.modern)
            finally:
                con.close()
            if english:
                try:
                    lex.english = import_module("12_ingest_writings").english_words()
                    lex.sources["english"] = len(lex.english)
                except Exception:                        # noqa: BLE001 - optional
                    lex.english = set()
        return lex

    @classmethod
    def from_words(cls, gurbani=(), modern=(), english=(), kosh=None, freq=None) -> "Lexicon":
        lex = cls()
        lex.gurbani, lex.modern, lex.english, lex.kosh = set(gurbani), set(modern), set(english), kosh
        lex.freq = Counter(freq or {})
        return lex

    def add_book_vocab(self, agreed_lines: list[list[str]], min_seen: int = MIN_SEEN, refuse=None) -> int:
        """
        agreed_lines: per line, the readings of every engine. A word counts
        once per line when at least two readings contain it. refuse(word):
        a word the caller knows to be a shared misreading is not admitted,
        whoever agrees on it.
        """
        counts: Counter = Counter()
        for readings in agreed_lines:
            sets = [set(words(r)) for r in readings if r]
            if len(sets) < 2:
                continue
            seen: Counter = Counter()
            for s in sets:
                seen.update(s)
            # two vowel signs on one letter is a misreading whoever agrees on it
            counts.update(w for w, n in seen.items() if n >= 2 and not _STACKED.search(w))
        self.book = {w for w, n in counts.items() if n >= min_seen and not (refuse and refuse(w))}
        self.sources["book"] = len(self.book)
        return len(self.book)

    # ------------------------------------------------------------------ queries
    def _clean(self, word: str) -> str:
        w = normalise(word).strip(_PUNCT)
        w, _ = strip_footnote_marker(w)
        return w

    def knows(self, word: str) -> bool:
        w = self._clean(word)
        if not w or _NUMERIC.match(w):
            return True
        if w in self.gurbani or w in self.modern or w in self.book:
            return True
        if "-" in w.strip("-"):
            # a compound (ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ, ਛਕਦਿਆਂ-ਛਕਦਿਆਂ) is known when its parts are
            return all(self.knows(p) for p in w.split("-") if p)
        if LATIN.search(w) and not GURMUKHI.search(w):
            return w.lower() in self.english
        if _STACKED.search(w):
            # two vowel signs on one letter (ਸੁੀ) is a misreading, not an
            # inflection: known only if a source spells it so (Gurbani's ਲੋੁੜੀਐ)
            return False
        if self.kosh is not None:
            entry, _ = self.kosh.lookup_word(w)
            if entry is not None:
                return True
        for suf in _MODERN_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 1:
                base = w[:-len(suf)]
                if base in self.modern or base in self.gurbani or base in self.book:
                    return True
                if self.kosh is not None and self.kosh.lookup_word(base)[0] is not None:
                    return True
        if _NUKTA in w:
            # the Kosh and Gurbani mostly write no nukta: ਪ੍ਰਸ਼ਾਦ is their ਪ੍ਰਸਾਦ
            return self.knows(w.replace(_NUKTA, ""))
        return False

    def oov(self, text: str) -> tuple[float, list[str]]:
        """(share of words not known, the unknown words), over words of >= 2 graphemes."""
        toks = [w for w in words(text) if len(graphemes(w)) >= 2]
        if not toks:
            return 0.0, []
        unknown = [w for w in toks if not self.knows(w)]
        return round(len(unknown) / len(toks), 3), unknown

    def candidates(self) -> set[str]:
        """Every word form the lexicon holds, for candidate generation."""
        out = set(self.gurbani) | self.modern | self.book
        if self.kosh is not None:
            out |= set(self.kosh.entries)
        return out
