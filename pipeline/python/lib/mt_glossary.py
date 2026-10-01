"""
A book's terms, as data: what the translation must call them, and the check that it did.

A 4B translation model given a Sikh biography rendered ਆਸਾ ਜੀ ਦੀ ਵਾਰ ਦਾ ਭੋਗ ਪਿਆ
("when the Asa di Var concluded") as "Asa Ji's time of death was near" and a
canopy for the Sri Guru Granth Sahib as one "for the Guru's funeral". Told the
terms in its prompt it did better, but explained them in brackets it was told
not to add, and some explanations were wrong (ਦਮੜਾ, a coin, "a type of drum").
So the terms are used three ways, all from one file per book (the manifest's
`glossary` key, a path beside manifest.json):

  prompt_lines()   the terms PRESENT in a paragraph, "ਕੜਾਹ ਪ੍ਰਸ਼ਾਦ = Karah
                   Prasad", for the prompt; a small model given forty pairs
                   drifts, so it is given the three it needs
  strip_glosses()  the model's emphasis and bracketed explanations taken out,
                   bounded by the source's own brackets (the author's
                   clarifications stay)
  missing()        a present term with none of its accepted English forms in
                   the translation: 26_translate_writings.check() rejects the
                   paragraph ("term: Karah Prasad") and the arbiter gets it

File format:

  {"rule": "Keep Sikh terms transliterated, not explained: ...",
   "terms": [{"pa": ["ਕੜਾਹ ਪ੍ਰਸ਼ਾਦ", "ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ"], "en": "Karah Prasad",
              "accept": ["Prasad", "Parshad"]}, ...]}

`pa` forms are compared after lib/ocr_text.normalise, whole words, with the
common inflections (ਸੰਗਤਾਂ is ਸੰਗਤ) unless the term says "whole": true (ਮਹਾਰਾਜ,
a saint's honorific, is not ਮਹਾਰਾਜਾ, a king); `en` is what the prompt asks for and is
always accepted; `accept` lists the other English forms that satisfy the check,
compared case-insensitively as whole words, a plural s allowed; `unless` lists
idioms in which the word is not the term (ਦੀ ਸੇਵਾ ਵਿੱਚ ਬੇਨਤੀ, "humbly submitted
to", is no Sewa), taken out of the source before the term is looked for.
`rule` is the one sentence of instruction the prompt carries before the terms.
"""
from __future__ import annotations
import json
import re

from lib.ocr_text import normalise

# endings a Punjabi noun takes that the glossary need not list
_INFLECT = ("", "ਾਂ", "ੇ", "ਿਆਂ", "ੀਆਂ", "ਾ", "ੀ", "ਂ", "ੋਂ")
_GURMUKHI_LETTER = "ਁ-ੵ"
_BRACKET = re.compile(r"\s*\(([^()]{1,120})\)")
_EMPHASIS = re.compile(r"(?<![\w*])[*_]{1,2}([^*_\n]{1,60}?)[*_]{1,2}(?![\w*])")


def load(path: str | None) -> dict:
    """{"rule": str, "terms": [...]} with every pa form normalised; empty when no path."""
    if not path:
        return {"rule": "", "terms": []}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    terms = []
    for i, t in enumerate(data.get("terms", [])):
        if not t.get("pa") or not t.get("en"):
            raise ValueError("%s: term %d needs 'pa' (a list) and 'en'" % (path, i))
        pa = [t["pa"]] if isinstance(t["pa"], str) else list(t["pa"])
        terms.append({"pa": [normalise(p) for p in pa if p.strip()], "en": t["en"].strip(),
                      "accept": [a.strip() for a in t.get("accept", []) if a.strip()],
                      "whole": bool(t.get("whole")),
                      "unless": [normalise(u) for u in t.get("unless", []) if u.strip()]})
    return {"rule": (data.get("rule") or "").strip(), "terms": terms}


def _pa_pattern(form: str, whole: bool = False) -> re.Pattern:
    ends = "|".join(re.escape(e) for e in sorted(("",) if whole else _INFLECT, key=len, reverse=True))
    return re.compile("(?<![%s])%s(?:%s)(?![%s])" % (_GURMUKHI_LETTER, re.escape(form), ends, _GURMUKHI_LETTER))


_PA_CACHE: dict[tuple, re.Pattern] = {}


def _occurs(form: str, text: str, whole: bool = False) -> bool:
    pat = _PA_CACHE.get((form, whole))
    if pat is None:
        pat = _PA_CACHE[(form, whole)] = _pa_pattern(form, whole)
    return bool(pat.search(text))


def present(glossary: dict, src: str) -> list[dict]:
    """The terms whose Punjabi occurs in src, outside the idioms each lists as `unless`."""
    s = normalise(src)
    out = []
    for t in glossary.get("terms", []):
        text = s
        for idiom in t.get("unless", []):
            text = text.replace(idiom, " ")
        if any(_occurs(p, text, t.get("whole", False)) for p in t["pa"]):
            out.append(t)
    return out


def accepted(term: dict, en: str) -> bool:
    for form in [term["en"]] + term["accept"]:
        if re.search(r"(?<![A-Za-z])%s(?:e?s)?(?![A-Za-z])" % re.escape(form), en, re.I):
            return True
    return False


def missing(glossary: dict, src: str, en: str) -> list[dict]:
    """Present terms that none of their accepted English forms renders."""
    return [t for t in present(glossary, src) if not accepted(t, en)]


_PLURAL = ("ਾਂ", "ਿਆਂ", "ੀਆਂ")


def inline_terms(glossary: dict, src: str) -> tuple[str, int]:
    """
    (source, n): each glossary term the source uses written in its English
    form inside the Punjabi, "ਆਸਾ ਜੀ ਦੀ ਵਾਰ ਦਾ ਭੋਗ ਪਿਆ" -> "Asa di Var ਦਾ bhog
    ਪਿਆ", for a translator that follows no instruction but keeps Latin words
    as they are (26 --prompt terms). Longer forms go first, so ਸਾਧ ਸੰਗਤ is
    one term and not ਸੰਗਤ; a plural ending gives an English plural.
    """
    s = normalise(src)
    n = 0
    forms = sorted(((p, t) for t in glossary.get("terms", []) for p in t["pa"]), key=lambda x: -len(x[0]))
    for form, t in forms:
        text_without = s
        for idiom in t.get("unless", []):
            text_without = text_without.replace(idiom, " ")
        if not _occurs(form, text_without, t.get("whole", False)):
            continue
        pat = _PA_CACHE.get((form, t.get("whole", False))) or _pa_pattern(form, t.get("whole", False))

        def repl(m, en=t["en"]):
            return en + ("s" if m.group(0).endswith(_PLURAL) else "")
        # an idiom listed as `unless` keeps its word
        guarded = s
        marks = {}
        for k, idiom in enumerate(t.get("unless", [])):
            key = "\u0000%d\u0000" % k
            if idiom in guarded:
                marks[key] = idiom
                guarded = guarded.replace(idiom, key)
        guarded, k = pat.subn(repl, guarded)
        for key, idiom in marks.items():
            guarded = guarded.replace(key, idiom)
        s, n = guarded, n + k
    return s, n


def prompt_lines(terms: list[dict], src: str = "") -> list[str]:
    """"ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ = Karah Prasad": each term in the form the paragraph uses, when given it."""
    s = normalise(src)
    out = []
    for t in terms:
        form = next((p for p in t["pa"] if s and _occurs(p, s, t.get("whole", False))), t["pa"][0])
        out.append("%s = %s" % (form, t["en"]))
    return out


def strip_glosses(en: str, src: str, terms: list[dict] | None = None) -> tuple[str, int]:
    """
    (text, n removed): *term* and _term_ become term; then, while the English
    has more brackets than the source, brackets go -- first one right after
    an accepted form of a present term (the model's "bhog (offering)"), then
    the shortest. The source's count is the bound, so the author's own
    clarifications (Darpan rule 4) survive in number.
    """
    out, n = _EMPHASIS.subn(r"\1", en)
    groups = list(_BRACKET.finditer(out))
    extra = len(groups) - src.count("(")
    if extra > 0:
        forms = [f for t in terms or [] for f in [t["en"]] + t["accept"]]
        after_term = re.compile(r"(?<![A-Za-z])(%s)(e?s)?$" % "|".join(
            re.escape(f) for f in sorted(forms, key=len, reverse=True)), re.I) if forms else None
        glossed = [m for m in groups if after_term and after_term.search(out[:m.start()].rstrip())]
        rest = sorted((m for m in groups if m not in glossed), key=lambda m: len(m.group(1)))
        drop = (glossed + rest)[:extra]
        pieces, last = [], 0
        for m in sorted(drop, key=lambda m: m.start()):
            pieces.append(out[last:m.start()])
            last = m.end()
        pieces.append(out[last:])
        out = "".join(pieces)
        n += len(drop)
    return re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s{2,}", " ", out)).strip(), n
