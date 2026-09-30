"""
Raags, taals and notation symbols: the names the keertan books print, keyed.

The data is lib/notation_vocab.json, one file for the pipeline and (as a
byte-identical mirror in packages/search-core) the web app. This module
loads it and answers three questions the parser asks on every heading:

  normalise_raag("ਰਾਗ ਸਾਰੰਗ (ਬਿੰਦ੍ਰਾਬਨੀ ਸਾਰੰਗ)")  -> key "brindavani_sarang", parent "sarang"
  normalise_taal("ਤਿੰਨਤਾਲ (ਮੱਧ ਲਯ)")               -> key "teentaal", laya "madh", 16 matras
  raag_key_from_corpus("Raag Sorath")             -> "sorath"   (what shabads.raag says)

A printed name is OCR output, so the match is layered: an alias after
folding (spelling variants the books use), then a consonant skeleton (ਤੋੜੀ
and ਟੋਡੀ are one raag), then a bounded edit distance. Each layer reports how
it decided, and a name no layer knows comes back as None rather than the
nearest guess: an unknown raag is a fact about the vocabulary, recorded by
the eval, not a wrong key in the database.

The corpus side never fuzzes. shabads.raag holds BaniDB's own spellings
(46 distinct values in the shipped gurbani.sqlite), every one of which is
listed under `banidb`; a gap is a build error.
"""
from __future__ import annotations
import json
import os
import re
import unicodedata

VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notation_vocab.json")

with open(VOCAB_PATH, encoding="utf-8") as _fh:
    VOCAB = json.load(_fh)

VERSION: str = VOCAB["version"]
RAAGS: dict[str, dict] = {r["key"]: r for r in VOCAB["raags"]}
TAALS: dict[str, dict] = {t["key"]: t for t in VOCAB["taals"]}
LAYA: list[dict] = VOCAB["laya"]
SECTIONS: dict[str, list[str]] = VOCAB["sections"]
SWARAS: dict = VOCAB["swaras"]
SYMBOLS: dict = VOCAB["symbols"]
ROMAN: dict = VOCAB["roman_map"]
STOP: dict[str, list[str]] = VOCAB["stopwords"]

SWARA_ORDER = tuple(SWARAS["order"])
KOMAL_ALLOWED = frozenset(SWARAS["komal_allowed"])
TIVRA_ALLOWED = frozenset(SWARAS["tivra_allowed"])

# Accept thresholds for the edit-distance layer: a raag name is longer and
# rarer than a taal name, so it may be held to a higher bar.
FUZZY_RAAG = 0.85
FUZZY_TAAL = 0.80
FUZZY_MARGIN = 0.05
FUZZY_MIN_LEN = 4

_GURMUKHI_LETTER = "ਅ-ਊਏਐਓ-ਨਪ-ਰਲਲ਼ਵਸ਼ਸਹਖ਼-ੜਫ਼ੲ-ੴ"
_VOWEL_SIGN = "ਾ-ੂੇੈੋੌ"
_KEEP_PA = re.compile("[^%s%sੰ ]" % (_GURMUKHI_LETTER, _VOWEL_SIGN))
_INDEPENDENT_VOWEL = re.compile("[ਅ-ਊਏਐਓਔੲ-ੴ]")
_VOWEL_SIGNS_RX = re.compile("[%sੰ]" % _VOWEL_SIGN)
_ZW = re.compile("[​‌‍­﻿]")
_WS = re.compile(r"\s+")
_FINAL_SHORT = re.compile("[ੁਿ]$")          # word-final ੁ / ਿ
_REPEAT = re.compile(r"(.)\1+")
_EN_KEEP = re.compile(r"[^a-z]+")
_EN_VOWEL = re.compile(r"[aeiou]")
# the skeleton folds the letters OCR and spelling most often swap
_SKEL_PA = str.maketrans({"ਟ": "ਤ", "ਠ": "ਥ", "ਡ": "ਦ", "ਢ": "ਧ",
                          "ੜ": "ਦ", "ਣ": "ਨ", "ਵ": "ਬ", "ਯ": "ਜ"})
_SKEL_EN = str.maketrans({"w": "v", "z": "j"})


def fold_pa(text: str, stop: str = "pa_raag") -> str:
    """
    The comparable form of a printed Gurmukhi name: NFC, nukta and addak
    dropped, bindi folded to tippi, subjoined letters flattened, everything
    that is not a letter or vowel sign turned to space, the words ਰਾਗ/ਮਹਲਾ/
    ਤਾਲ... removed, and a word-final ੁ or ਿ dropped (ਬਿਲਾਵਲੁ and ਬਿਲਾਵਲ are
    one word). Idempotent.
    """
    s = unicodedata.normalize("NFC", text or "")
    s = _ZW.sub("", s)
    s = s.replace("਼", "").replace("ੱ", "").replace("ਂ", "ੰ").replace("੍", "")
    s = _KEEP_PA.sub(" ", s)
    stops = {_fold_pa_token(w) for w in STOP.get(stop, [])}
    out = []
    for tok in s.split():
        tok = _fold_pa_token(tok)
        if not tok or tok in stops:
            continue
        out.append(tok)
    return " ".join(out)


def _fold_pa_token(tok: str) -> str:
    tok = tok.replace("਼", "").replace("ੱ", "").replace("ਂ", "ੰ").replace("੍", "")
    return _FINAL_SHORT.sub("", tok)


def skel_pa(folded: str) -> str:
    """Consonant skeleton of a folded name: ਤੋੜੀ, ਟੋਡੀ and ਤੋਡੀ all become ਤਦ."""
    s = _VOWEL_SIGNS_RX.sub("", folded)
    s = _INDEPENDENT_VOWEL.sub("", s)
    s = s.replace(" ", "").translate(_SKEL_PA)
    return _REPEAT.sub(r"\1", s)


def fold_en(text: str, stop: str = "en_raag") -> str:
    """Lowercase ASCII words of an English name, without diacritics or the word raag."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = _EN_KEEP.sub(" ", s)
    stops = set(STOP.get(stop, []))
    return " ".join(w for w in s.split() if w not in stops)


def skel_en(folded: str) -> str:
    """As translit.js skelKey: vowels out, w/v and z/j folded, doubles collapsed."""
    s = folded.replace(" ", "").translate(_SKEL_EN)
    s = _EN_VOWEL.sub("", s)
    return _REPEAT.sub(r"\1", s)


def indel_ratio(a: str, b: str) -> float:
    """1 - (|a|+|b|-2·LCS)/(|a|+|b|): rapidfuzz's normalized Indel similarity, in plain Python."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    return 1.0 - (len(a) + len(b) - 2 * lcs) / (len(a) + len(b))


class _Index:
    """Alias, skeleton and fuzzy lookups over one family (raags or taals), one script."""

    def __init__(self, entries: dict[str, dict], script: str, stop: str):
        self.script, self.stop = script, stop
        self.alias: dict[str, str] = {}
        self.alias_nospace: dict[str, str] = {}
        skel: dict[str, set[str]] = {}
        self.names: list[tuple[str, str]] = []          # (folded alias, key), for the fuzzy layer
        for key, e in entries.items():
            spellings = list(e["aliases"].get(script, []))
            if script == "en":
                spellings += [e["en"], e.get("banidb") or ""]
            else:
                spellings.append(e.get("pa") or "")
            for sp in spellings:
                f = self.fold(sp)
                if not f:
                    continue
                self.alias.setdefault(f, key)
                self.alias_nospace.setdefault(f.replace(" ", ""), key)
                self.names.append((f, key))
                skel.setdefault(self.skel(f), set()).add(key)
        # a skeleton two keys share decides nothing
        self.skel_index = {s: next(iter(ks)) for s, ks in skel.items() if len(ks) == 1}
        self.collisions = {s: sorted(ks) for s, ks in skel.items() if len(ks) > 1}

    def fold(self, s: str) -> str:
        return fold_pa(s, self.stop) if self.script == "pa" else fold_en(s, self.stop)

    def skel(self, folded: str) -> str:
        return skel_pa(folded) if self.script == "pa" else skel_en(folded)

    def lookup(self, text: str, fuzzy: float | None) -> dict:
        folded = self.fold(text)
        if not folded:
            return {"key": None, "confidence": 0.0, "method": None}
        key = self.alias.get(folded) or self.alias_nospace.get(folded.replace(" ", ""))
        if key:
            return {"key": key, "confidence": 1.0, "method": "alias"}
        sk = self.skel(folded)
        # a skeleton of two letters says too little: ਨੋਟ would be Nat, ਸ would be Asa
        if len(sk) >= 3 and sk in self.skel_index:
            return {"key": self.skel_index[sk], "confidence": 0.9, "method": "skeleton"}
        if fuzzy is not None and len(folded.replace(" ", "")) >= FUZZY_MIN_LEN:
            best = self._fuzzy(folded, fuzzy)
            if best:
                return best
        # leftover noise around an OCR'd name (ਮਹਲਾ ੫ ਘਰੁ ੩, "ਸੂਹੀ ਵਿਚ ਕਾਫੀ"):
        # the longest run of tokens that is a name on its own. Not for corpus
        # strings (fuzzy is None there): "Guru Arjan Dev Ji" must stay nothing.
        toks = folded.split()
        if fuzzy is not None and len(toks) >= 2:
            for n in range(len(toks) - 1, 0, -1):
                for i in range(0, len(toks) - n + 1):
                    sub = " ".join(toks[i:i + n])
                    sk = self.skel(sub)
                    key = self.alias.get(sub)
                    if not key and len(sk) >= 3:
                        key = self.skel_index.get(sk)
                    if key:
                        return {"key": key, "confidence": 0.8, "method": "partial"}
        return {"key": None, "confidence": 0.0, "method": None}

    def _fuzzy(self, folded: str, accept: float) -> dict | None:
        scored = sorted(((indel_ratio(folded, name), key) for name, key in self.names), reverse=True)
        if not scored or scored[0][0] < accept:
            return None
        best, key = scored[0]
        other = next((s for s, k in scored if k != key), 0.0)
        if best - other < FUZZY_MARGIN:
            return None
        return {"key": key, "confidence": round(best, 3), "method": "fuzzy"}


_RAAG_PA = _Index(RAAGS, "pa", "pa_raag")
_RAAG_EN = _Index(RAAGS, "en", "en_raag")
_TAAL_PA = _Index(TAALS, "pa", "pa_taal")
_TAAL_EN = _Index(TAALS, "en", "en_taal")
_BANIDB = {r["banidb"]: r["key"] for r in RAAGS.values() if r.get("banidb")}
_GURMUKHI_RX = re.compile("[਀-੿]")


def _script(text: str) -> str:
    return "pa" if _GURMUKHI_RX.search(text or "") else "en"


def normalise_raag(text: str, script: str | None = None) -> dict:
    """
    @returns {"key", "parent", "confidence", "method"}; key None when unknown.
    The parenthesised variant a heading adds -- "ਸਾਰੰਗ (ਬਿੰਦ੍ਰਾਬਨੀ ਸਾਰੰਗ)" --
    is the more specific name, so it is tried first and the bare name is the
    fallback and the parent.
    """
    script = script or _script(text)
    index = _RAAG_PA if script == "pa" else _RAAG_EN
    fuzzy = FUZZY_RAAG if script == "pa" else None
    inner = re.findall(r"\(([^()]+)\)", text or "")
    outer = re.sub(r"\([^()]*\)", " ", text or "")
    hit = None
    for cand in inner + [outer]:
        r = index.lookup(cand, fuzzy)
        if r["key"]:
            hit = r
            break
    if not hit:
        hit = index.lookup(text or "", fuzzy)
    key = hit["key"]
    parent = RAAGS[key].get("parent") if key else None
    if key and inner:
        # the bare name outside the brackets names the parent when the table does not
        base = index.lookup(outer, None)["key"]
        if base and base != key and not parent:
            parent = base
    return {"key": key, "parent": parent, "confidence": hit["confidence"], "method": hit["method"]}


def normalise_taal(text: str, script: str | None = None) -> dict:
    """@returns {"key", "laya", "matras", "confidence", "method"}."""
    script = script or _script(text)
    laya, rest = split_laya(text or "", script)
    index = _TAAL_PA if script == "pa" else _TAAL_EN
    hit = index.lookup(rest, FUZZY_TAAL if script == "pa" else None)
    key = hit["key"]
    return {"key": key, "laya": laya, "matras": TAALS[key]["matras"] if key else None,
            "confidence": hit["confidence"], "method": hit["method"]}


def split_laya(text: str, script: str = "pa") -> tuple[str | None, str]:
    """('madh', 'ਤਿੰਨਤਾਲ') from 'ਤਿੰਨਤਾਲ (ਮੱਧ ਲਯ)': the laya named, and the text without it."""
    found = None
    rest = text
    for entry in LAYA:
        for alias in sorted(entry[script if script in ("pa", "en") else "pa"], key=len, reverse=True):
            pattern = re.compile(r"(?<![\u0a00-\u0a7fA-Za-z])\(?\s*" + re.escape(alias) + r"\s*\)?(?![\u0a00-\u0a7fA-Za-z])", re.I)
            if pattern.search(rest):
                found = entry["key"]
                rest = pattern.sub(" ", rest)
    return found, rest


def raag_key_from_corpus(name: str | None) -> str | None:
    """The key of a shabads.raag value: BaniDB's exact spelling first, then the English aliases."""
    if not name:
        return None
    if name in _BANIDB:
        return _BANIDB[name]
    return _RAAG_EN.lookup(name, None)["key"]


def taal_info(key: str | None) -> dict | None:
    return TAALS.get(key) if key else None


def taal_markers(key: str, matra_from: int = 1, count: int | None = None) -> list[str | None]:
    """
    The marker printed over each matra of an avartan (None where there is
    none): teentaal -> ['×', None, None, None, '2', ...]. `matra_from` and
    `count` cut the list to the matras a printed row actually shows.
    """
    t = TAALS.get(key)
    if not t or not t.get("vibhag"):
        return [None] * (count or 0)
    marks: list[str | None] = []
    for i, size in enumerate(t["vibhag"]):
        marks.append(t["markers"][i] if t.get("markers") and i < len(t["markers"]) else None)
        marks.extend([None] * (size - 1))
    start = max(0, matra_from - 1)
    end = start + count if count is not None else len(marks)
    return marks[start:end]


def taal_from_markers(n_beats: int, markers: dict[int, str]) -> str | None:
    """
    The taal whose matra count and sam/khali positions match what a marker
    row printed: 10 beats with × at 1 and 0 at 6 is jhaptaal. None when no
    taal matches or two do.
    """
    hits = []
    for key, t in TAALS.items():
        if t.get("matras") != n_beats or not t.get("vibhag"):
            continue
        ok = True
        for m, mark in markers.items():
            kind = marker_kind(mark)
            if kind == "sam" and m != t["sam"]:
                ok = False
            elif kind == "khali" and m not in t["khali"]:
                ok = False
            elif kind == "tali" and m not in t["tali"]:
                ok = False
        if ok:
            hits.append(key)
    return hits[0] if len(hits) == 1 else None


def marker_kind(glyph: str) -> str | None:
    """'sam' | 'khali' | 'tali' | None for a printed marker glyph."""
    g = (glyph or "").strip()
    if g in SYMBOLS["sam"]:
        return "sam"
    if g in SYMBOLS["khali"]:
        return "khali"
    for glyphs in SYMBOLS["tali"].values():
        if g in glyphs:
            return "tali"
    return None


def section_label(text: str) -> tuple[str, int | None] | None:
    """('antara', 2) from 'ਅੰਤਰਾ ੨' or 'ਅੰਤਰਾ 2:'; ('sthai', None) from 'ਸਥਾਈ'; None otherwise."""
    s = (text or "").strip()
    low = s.lower()
    for kind, labels in SECTIONS.items():
        for label in labels:
            if low.startswith(label.rstrip(": -").lower()):
                tail = s[len(label.rstrip(": -")):]
                m = re.search(r"([0-9੦-੯]+)", tail)
                n = int(gurmukhi_digits(m.group(1))) if m else None
                if len(tail.strip(" :-.()")) > 6 and not m:
                    return None                  # a sentence that begins with the word, not a label
                return kind, n
    return None


def gurmukhi_digits(s: str) -> str:
    """'੧੨੦੨' -> '1202'; ASCII digits pass through."""
    return "".join(str(ord(c) - 0x0a66) if "੦" <= c <= "੯" else c for c in s or "")


def swara_read(token: str, script: str = "gurmukhi") -> str | None:
    """The canonical swara of one printed cell token ('ਸਾ' -> 'S', 'n' -> 'N'), or None."""
    t = (token or "").strip()
    if not t:
        return None
    if script == "gurmukhi":
        t = t.replace("਼", "")
        return SWARAS["gurmukhi_read"].get(t)
    if script == "devanagari":
        return SWARAS["devanagari_read"].get(t)
    u = t.upper()
    return u if u in SWARA_ORDER and len(t) == 1 else None
