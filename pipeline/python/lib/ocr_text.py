"""
Gurmukhi text as OCR engines emit it, made comparable.

Three engines reading the same line return three different byte sequences for
what a reader would call the same text, and a ground-truth typist produces a
fourth. Everything that compares OCR to OCR, OCR to the corpus, or OCR to
ground truth goes through normalise() first, applied identically to BOTH
sides, so that a difference means a reading error and not an encoding choice.

What normalise() does, in order:

  NFC        Unicode composition. Note the direction it takes the nukta
             letters: ਸ਼ ਖ਼ ਗ਼ ਜ਼ ਫ਼ ਲ਼ are composition EXCLUSIONS, so NFC turns the
             precomposed letter into base + U+0A3C. Both sides get it, so the
             comparison holds; nothing here ever writes text back to a reader.
  carriers   ੳ ਅ ੲ with a vowel sign are how a typist or an engine sometimes
             spells the independent vowels ਉ ਊ ਓ ਆ ਐ ਔ ਇ ਈ ਏ. Unicode does NOT
             make these canonically equivalent, so a table does it.
  sihari     ਿ is drawn BEFORE the consonant it follows. An engine that emits
             glyph order puts it first; logical order puts it second. Only the
             unambiguous cases are moved here: a sihari at the start of a word
             or after a non-letter can only be visual order. A sihari between
             two consonants is ambiguous (ਸਿਮ vs a visual-order ਿਸਮ) and is left
             to the engine-specific fix (gurmukhifix, run only on Tesseract).
  marks      a combining mark repeated by a stutter is collapsed; ZWJ/ZWNJ and
             soft hyphens are dropped; whitespace is collapsed.

What it deliberately does not do: fold tippi and bindi, or rewrite addak. Both
are orthographically meaningful in Gurbani and a normaliser that "fixes" them
(indic_nlp_library's canonicalisers do) destroys the text it is meant to
protect. match_key() folds ਂ/ੰ and drops the nukta -- but that key is used only
to FIND a corpus line, never as the text that replaces it.

Graphemes, not code points, are the unit of comparison: a dropped matra is one
error, not a broken cluster, and a two-code-point nukta letter is one symbol.
"""
from __future__ import annotations
import re
import unicodedata

try:                                   # \X: extended grapheme clusters
    import regex as _regex
    _GRAPHEME = _regex.compile(r"\X")
except ImportError:                    # a small clusterer of our own, below
    _regex = None
    _GRAPHEME = None

GURMUKHI = re.compile("[\u0a00-\u0a7f]")
DEVANAGARI = re.compile("[\u0900-\u097f]")
LATIN = re.compile("[A-Za-z]")
GURMUKHI_DIGIT = re.compile("[\u0a66-\u0a6f]")

# Consonants that can carry a vowel sign (the letters proper, plus the nukta forms).
_CONSONANT = "\u0a15-\u0a39\u0a59-\u0a5e"
_NUKTA = "\u0a3c"
_VOWEL_SIGNS = "\u0a3e-\u0a4c"                 # ਾ ਿ ੀ ੁ ੂ ੇ ੈ ੋ ੌ
_COMBINING = "\u0a01\u0a02\u0a03\u0a3c\u0a3e-\u0a4d\u0a70\u0a71\u0a75"

# Independent vowels written as carrier + sign. Not canonically equivalent, so a table.
CARRIERS = {
    "\u0a73\u0a41": "\u0a09",   # ੳ + ੁ -> ਉ
    "\u0a73\u0a42": "\u0a0a",   # ੳ + ੂ -> ਊ
    "\u0a73\u0a4b": "\u0a13",   # ੳ + ੋ -> ਓ
    "\u0a05\u0a3e": "\u0a06",   # ਅ + ਾ -> ਆ
    "\u0a05\u0a48": "\u0a10",   # ਅ + ੈ -> ਐ
    "\u0a05\u0a4c": "\u0a14",   # ਅ + ੌ -> ਔ
    "\u0a72\u0a3f": "\u0a07",   # ੲ + ਿ -> ਇ
    "\u0a72\u0a40": "\u0a08",   # ੲ + ੀ -> ਈ
    "\u0a72\u0a47": "\u0a0f",   # ੲ + ੇ -> ਏ
}
_CARRIER_RX = re.compile("|".join(re.escape(k) for k in CARRIERS))

# A sihari that can only be in visual order: at the start of a word (or after
# anything that is not a Gurmukhi letter) and followed by a consonant.
_SIHARI_LEADING = re.compile("(?<![%s%s%s])\u0a3f([%s]%s?)" % (_CONSONANT, _VOWEL_SIGNS, _NUKTA, _CONSONANT, _NUKTA))
_DUP_MARK = re.compile("([%s])\\1+" % _COMBINING)
_DROP = re.compile("[\u200b\u200c\u200d\u00ad\ufeff]")
_WS = re.compile(r"\s+")
# Quotation glyphs: the books set \u2018 \u2019 \u201c \u201d, engines emit any of ' ` \u00b4 " and
# the typist types ASCII. Which glyph was used is not a reading error.
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "`": "'", "\u00b4": "'", "\u201c": '"', "\u201d": '"',
                         "\u2032": "'", "\u2033": '"'})

# Punctuation and structure that carries no reading: dandas, counters, brackets.
_NON_KEY = re.compile("[^%s%sA-Za-z0-9\u0900-\u097f]" % ("\u0a01-\u0a75", ""))
_DANDA = re.compile("[\u0964\u0965|]")


def normalise(text: str) -> str:
    """The comparable form of a line. Idempotent."""
    s = unicodedata.normalize("NFC", text or "")
    s = _DROP.sub("", s).translate(_QUOTES)
    s = _CARRIER_RX.sub(lambda m: CARRIERS[m.group(0)], s)
    s = _SIHARI_LEADING.sub("\\1\u0a3f", s)
    s = _DUP_MARK.sub("\\1", s)
    return _WS.sub(" ", s).strip()


def graphemes(text: str) -> list[str]:
    """Extended grapheme clusters; a consonant with its marks is one item."""
    if _GRAPHEME is not None:
        return [g for g in _GRAPHEME.findall(text) if g]
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        joins = cat in ("Mn", "Mc", "Me") or (out and out[-1].endswith("\u0a4d"))
        if joins and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def script_of(text: str) -> str:
    """"gurmukhi", "latin", "devanagari", "mixed" or "none", by letters present."""
    g, d, l = bool(GURMUKHI.search(text)), bool(DEVANAGARI.search(text)), bool(LATIN.search(text))
    n = sum((g, d, l))
    if n == 0:
        return "none"
    if n > 1:
        return "mixed"
    return "gurmukhi" if g else ("devanagari" if d else "latin")


def match_key(text: str) -> list[str]:
    """
    Graphemes of a line for FINDING it in the corpus. Retrieval-only folds:
    dandas, counters, punctuation and spaces are dropped, ਂ is folded to ੰ and
    the nukta is dropped, because those are the marks engines most often get
    wrong and a match must survive them. The corpus text itself is never
    changed by this.
    """
    from lib.gurmukhi_text import clean_gurmukhi        # strips ॥੧॥, ॥ ਰਹਾਉ ॥ as whole segments
    s = clean_gurmukhi(normalise(text))
    s = _DANDA.sub(" ", s)
    # BaniDB writes the subjoined ha of \u0a06\u0a2a\u0a40\u0a28\u0a4d\u0a39\u0a48 with U+0A51 (udaat); the books
    # print virama + ha. One form for finding, whichever the source used.
    s = s.replace("\u0a4d\u0a39", "\u0a51")
    s = s.replace("\u0a02", "\u0a70").replace("\u0a3c", "")
    s = _NON_KEY.sub("", s)
    return graphemes(s)


def words(text: str) -> list[str]:
    """Whitespace tokens with surrounding punctuation stripped; empties dropped."""
    out = []
    for tok in normalise(text).split(" "):
        tok = tok.strip(".,;:!?\"'()[]{}\u0964\u0965|-\u2013\u2014\u2018\u2019\u201c\u201d")
        if tok:
            out.append(tok)
    return out


_FOOTNOTE_MARK = re.compile("^(.{2,}?[^0-9\u0a66-\u0a6f])([0-9\u0a66-\u0a6f]{1,2})$")


def strip_footnote_marker(word: str):
    """
    'ਸ਼ਬਦ੧' -> ('ਸ਼ਬਦ', '੧'): a superscript footnote number an engine glued to the
    word before it. Only a one- or two-digit tail on a word of letters counts;
    a bare number, a year or a verse counter is left alone.
    """
    m = _FOOTNOTE_MARK.match(word or "")
    if not m or not (GURMUKHI.search(m.group(1)) or LATIN.search(m.group(1))):
        return word, None
    return m.group(1), m.group(2)


def _levenshtein(a: list, b: list) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Grapheme error rate: edits / reference graphemes, both sides normalised."""
    r, h = graphemes(normalise(ref)), graphemes(normalise(hyp))
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def wer(ref: str, hyp: str) -> float:
    """Word error rate on punctuation-stripped tokens."""
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


_COUNTER = re.compile(r"^[੦-੯0-9\s]+$")
_RAHAO = re.compile(r"^\s*ਰਹਾਉ(\s+ਦੂਜਾ)?\s*$")


def key_positions(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    match_key() with a map back: (key graphemes, [(start, end) in normalise(text)]).

    Same folds as match_key -- dandas, counters and the rahao marker dropped as
    whole segments, punctuation and spaces dropped, nasal and nukta folded --
    but each kept grapheme remembers where it sits in the normalised line, so
    a matched span can be replaced by corpus text without losing the citation
    that follows it.
    """
    n = normalise(text)
    # danda-delimited segments that carry no reading: counters, rahao marker
    skip = [False] * len(n)
    pos = 0
    for seg in re.split("([।॥|])", n):
        if seg in ("।", "॥", "|"):
            skip[pos] = True
            pos += 1
            continue
        stripped = seg.strip()
        if not stripped or _COUNTER.match(stripped) or _RAHAO.match(stripped):
            for i in range(pos, pos + len(seg)):
                skip[i] = True
        pos += len(seg)
    key: list[str] = []
    spans: list[tuple[int, int]] = []
    i = 0
    for g in graphemes(n):
        start, end = i, i + len(g)
        i = end
        if skip[start]:
            continue
        folded = g.replace("੍ਹ", "ੑ").replace("ਂ", "ੰ").replace("਼", "")
        folded = _NON_KEY.sub("", folded)
        if not folded:
            continue
        key.append(folded)
        spans.append((start, end))
    return key, spans


def splice(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """
    normalise(text) with [start, end) character spans replaced, in order.
    Spans come from key_positions(); the text between them (a citation, a
    stray word) is kept as read.
    """
    n = normalise(text)
    out, cur = [], 0
    for start, end, new in sorted(replacements):
        start, end = max(start, cur), max(end, cur)
        out.append(n[cur:start])
        out.append(new)
        cur = end
    out.append(n[cur:])
    return _WS.sub(" ", "".join(out)).strip()
