"""
The text of a notation page, read: references, headings, cell tokens.

Everything here works on strings the OCR produced and nothing on pixels.
Three questions:

  parse_ref("(ਸਾਰੰਗ ਮ: ੫, ੧੨੦੨)")        where in the Granth the printed shabad is:
                                       ang 1202, mahala 5, raag sarang, source G
  parse_heading("੪. ਰਾਗ ਸਾਰੰਗ (ਬਿੰਦ੍ਰਾਬਨੀ ਸਾਰੰਗ), ਤਿੰਨਤਾਲ (ਮੱਧ ਲਯ)")
                                       the raag the composer used, the taal,
                                       the laya, a section label, a reet note
  swara_token("ਸਾ") / clean_bol("ਤਰਿਆऽ") / token_class(...)
                                       what one cell of a grid row says

The books print an ang in eight ways (docs/notations.md lists them); every
form is a bracket or a trailing phrase with a number in Gurmukhi or Arabic
digits, so the parser looks for the number and reads the words around it
for the source and the raag. A bare bracketed number is an ang only when it
is two digits or more, or a keyword stands beside it: (੧) after a word is a
footnote.
"""
from __future__ import annotations
import re

from lib.notation_vocab import (LAYA, SYMBOLS, gurmukhi_digits, marker_kind, normalise_raag, normalise_taal,
                                section_label, split_laya, swara_read)

DIGITS = "0-9੦-੯"
_NUM = re.compile("[%s]+" % DIGITS)
_RANGE = re.compile("([%s]{1,4})\\s*[-–—]\\s*([%s]{1,4})" % (DIGITS, DIGITS))
_BRACKET = re.compile(r"[\(\[]([^()\[\]]{1,80})[\)\]]")
_MAHALA = re.compile("(?:ਮਹਲਾ|ਮਹੱਲਾ|ਮ[:ਃ]|ਮਃ|M\\.?|ਪਾਤਸ਼ਾਹੀ|ਪਾਤਿਸ਼ਾਹੀ|ਪਾ[:ਃ.])\\s*([%s]{1,2})" % DIGITS)   # a mahala or a patshahi: not an ang
_ANG_WORDS = re.compile(r"ਪੰਨਾ|ਪੰਨੇ|ਅੰਗ|ਪਨਾ|SGGS|S\.G\.G\.S|ਗੁ\.?\s*ਗ੍ਰੰ|ਗ੍ਰੰਥ|ਆਦਿ ਗ੍ਰੰਥ|\bp\.|\bpage\b|\bang\b", re.I)
_SOURCE_WORDS = (
    ("B", re.compile(r"ਭਾਈ ਗੁਰਦਾਸ|ਵਾਰਾਂ ਭਾਈ|ਪਉੜੀ|ਵਾਰ\s+[%s]{1,2}\b" % DIGITS)),
    ("K", re.compile(r"ਕਬਿੱਤ|ਕਬਿਤ|ਸਵੱਯੇ ਭਾਈ")),
    ("D", re.compile(r"ਦਸਮ|ਜਾਪੁ ਸਾਹਿਬ|ਅਕਾਲ ਉਸਤਤਿ|ਚੌਪਈ|ਸਵੱਯੇ ਪਾਤਸ਼ਾਹੀ|ਪਾਤਸ਼ਾਹੀ ੧੦|ਪਾ: ੧੦")),
    ("N", re.compile(r"ਨੰਦ ਲਾਲ|ਗੋਯਾ|ਗ਼ਜ਼ਲ")),
)
MAX_ANG = 1430
_HEADING_WORDS = re.compile(r"(?:^|[\s(:\-—])(?:ਰਾਗ|ਰਾਗੁ|ਤਾਲ|ਤਾਲਾ|ਮਾਤਰਾ|ਮਾਤਰਾਂ|ਮਾਤ੍ਰਾ|ਨੋਟੀਸ਼ਨ)(?=[\s:,\-—)]|$)")
_NOTE_WORDS = re.compile(r"^\s*[\-—–']*\s*(?:ਨੋਟ|ਨੌਟ|ਨੇਂਟ|ਨੋਂਟ)|ਬਾਕੀ ਤੁਕਾਂ|ਅੰਤਰੇ ਤੇ ਲਾਓ|ਤੇ ਲਾਓ|ਦੀ ਰੀਤ|ਦੀ ਤਰਜ਼|ਵਾਂਗ ਗਾਓ|ਲਿਖਿਆ ਹੈ")
_HELD_LIKE = re.compile(r"^[Ss$5ऽ਽;,.]+$")
_REET = re.compile(r"(?:ਦੀ|ਵਾਲੀ)?\s*(?:ਰੀਤ|ਤਰਜ਼|ਤਰਜ|ਧਾਰਨਾ|ਧੁਨ)")
_NUMBER_PREFIX = re.compile("^\\s*([%s]{1,3})\\s*[.)।:-]\\s*" % DIGITS)
_MATRAS = re.compile("(?:ਮਾਤਰਾ|ਮਾਤਰਾਂ|ਮਾਤ੍ਰਾ|ਮਾਤਰੇ|matras?)\\s*[-:]?\\s*([%s]{1,2})" % DIGITS, re.I)
_PUNCT = "()[]{}.,;:!?'\"‘’“”।॥|"
_EXT_MARKS = set(SYMBOLS["extension"]) - {"S", "ਸ", "."}
_OCTAVE_SIGNS = re.compile("[ੁੂੰਂੑ]")
_BELOW_SIGNS = re.compile("[ੁੂੑ]")
_ABOVE_SIGNS = re.compile("[ੰਂ]")
_HELD_TAIL = re.compile("[ऽ਽S5ਸ]+$")
# a number that counts something other than a page: the vaar, the pauri, the
# kabitt, the swaiyya, the chhand, the salok, the ashtpadi
_NOT_ANG = re.compile("(?:ਵਾਰ|ਪਉੜੀ|ਪਉੜੀਆਂ|ਕਬਿੱਤ|ਕਬਿਤ|ਸਵੱਯੇ|ਸਵਈਏ|ਸਵੈਯੇ|ਛੰਦ|ਸਲੋਕ|ਸ਼ਲੋਕ|ਅਸ਼ਟਪਦੀ|ਅਸਟਪਦੀ|ਪਦਾ|ਪਦੇ|ਨੰ[:ਃ.]?)\s*[%s]{1,3}" % DIGITS)


def _int(s: str) -> int:
    return int(gurmukhi_digits(s))


def _complete_range(a: int, b_txt: str) -> int:
    """959-60 -> 960: a short second number repeats the first's leading digits."""
    b = _int(b_txt)
    digits = len(gurmukhi_digits(b_txt))
    if b < a and digits < len(str(a)):
        b = int(str(a)[: len(str(a)) - digits] + gurmukhi_digits(b_txt))
    return b


def parse_ref(text: str) -> dict | None:
    """
    The printed reference in a line, or None.

    @returns {"text", "ang_from", "ang_to", "source", "raag_key", "mahala", "span": (start, end)}
    `source` is G unless the words name another scripture; `ang_from` is
    None when the reference names a source but no number (a Vaar cited by
    pauri, say).
    """
    if not text:
        return None
    best = None
    for m in _BRACKET.finditer(text):
        got = _read_ref(m.group(1), bracketed=True)
        if got:
            got["text"] = m.group(0)
            got["span"] = (m.start(), m.end())
            best = got if best is None or (got["ang_from"] and not best["ang_from"]) else best
    if best:
        return best
    # no bracket: a trailing phrase such as "ਪੰਨਾ ੭੦੦" or "SGGS 679"
    m = _ANG_WORDS.search(text)
    if m:
        tail = text[m.start():]
        got = _read_ref(tail, bracketed=False)
        if got and got["ang_from"]:
            got["text"] = tail.strip()
            got["span"] = (m.start(), len(text))
            return got
    return None


def _read_ref(inner: str, bracketed: bool) -> dict | None:
    source = "G"
    for code, rx in _SOURCE_WORDS:
        if rx.search(inner):
            source = code
            break
    mah = _MAHALA.search(inner)
    mahala = _int(mah.group(1)) if mah else None
    has_word = bool(_ANG_WORDS.search(inner))
    ang_from = ang_to = None
    rng = _RANGE.search(inner)
    if rng:
        a = _int(rng.group(1))
        ang_from, ang_to = a, _complete_range(a, rng.group(2))
    else:
        # the last number that is not the mahala's
        nums = [(m.start(), m.group(0)) for m in _NUM.finditer(inner)]
        taken = [m.span() for m in _NOT_ANG.finditer(inner)] + ([mah.span()] if mah else [])
        nums = [(s, t) for s, t in nums if not any(a <= s < b for a, b in taken)]
        if nums:
            ang_from = ang_to = _int(nums[-1][1])
    limit = 40 if source == "B" else MAX_ANG
    if ang_from is not None and not (1 <= ang_from <= limit and ang_from <= ang_to <= limit):
        ang_from = ang_to = None
    # a bare number in brackets is a footnote unless it is long or has company
    if ang_from is not None and bracketed and not has_word and mahala is None and source == "G":
        only_number = _NUM.sub("", inner).strip(" ,.-–—") == ""
        if only_number and ang_from < 10:
            return None
    raag_key = None
    if inner and not (bracketed and _NUM.sub("", inner).strip(" ,.-") == ""):
        words = _NUM.sub(" ", _MAHALA.sub(" ", inner))
        words = _ANG_WORDS.sub(" ", words)
        r = normalise_raag(words)
        raag_key = r["key"] if r["method"] in ("alias", "skeleton") else None
    if ang_from is None and not has_word and source == "G" and raag_key is None:
        return None
    return {"ang_from": ang_from, "ang_to": ang_to, "source": source, "raag_key": raag_key, "mahala": mahala}


def is_note_like(text: str) -> bool:
    """An instruction to the singer, not a heading: 'ਨੋਟ:- ...', 'ਬਾਕੀ ਤੁਕਾਂ ਅੰਤਰੇ ਤੇ ਲਾਓ'."""
    return bool(_NOTE_WORDS.search((text or "").strip()))


def is_heading_like(text: str) -> bool:
    """
    A short line that names a raag or a taal the way a heading does (the
    word ਰਾਗ/ਤਾਲ beside it, or nothing but the name), or a section label. A
    verse that happens to contain a raag's name is not one.
    """
    t = (text or "").strip()
    if not t or len(t.split()) > 14 or is_note_like(t):
        return False
    if section_label(t):
        return True
    if _HEADING_WORDS.search(t) and len(t.split()) <= 8:      # a sentence of prose that mentions a raag is not one
        return True
    if len(t.split()) <= 4:
        return normalise_taal(t)["method"] == "alias" or normalise_raag(t)["method"] == "alias"
    return False


def parse_heading(text: str) -> dict:
    """
    {"raw", "number", "raag": {"printed", "key", "parent_key", "confidence"} | None,
     "taal": {"printed", "key", "matras", "laya", "confidence"} | None,
     "laya", "section": {"kind", "n"} | None, "reet_of": str | None}
    """
    raw = (text or "").strip()
    s = raw
    number = None
    m = _NUMBER_PREFIX.match(s)
    if m:
        number = _int(m.group(1))
        s = s[m.end():]
    section = None
    sec = section_label(s)
    if sec:
        section = {"kind": sec[0], "n": sec[1]}
    reet_of = None
    rm = _REET.search(s)
    if rm:
        # "ਅਉਖੀ ਘੜੀ ਨ ਦੇਖਣ ਦੇਈ ਦੀ ਰੀਤ ਤੇ": what precedes the word is the tune
        before = s[:rm.start()].strip(" :-‘’'\"")
        after = s[rm.end():].strip(" :-‘’'\"")
        quoted = re.search(r"[‘'\"]([^’'\"]+)[’'\"]", s)
        reet_of = quoted.group(1) if quoted else (before.split(",")[-1].strip() or after or None)
    laya, rest = split_laya(s)
    matras_m = _MATRAS.search(rest)
    printed_matras = _int(matras_m.group(1)) if matras_m else None
    taal = _find_taal(rest)
    taal_span = taal.pop("span") if taal else None
    remainder = rest if not taal_span else rest[:taal_span[0]] + " " + rest[taal_span[1]:]
    if matras_m:
        remainder = remainder.replace(matras_m.group(0), " ")
    if section:
        remainder = _strip_section(remainder)
    raag = None
    r = normalise_raag(remainder) if remainder.strip() else {"key": None, "parent": None, "confidence": 0.0, "method": None}
    if r["key"]:
        printed = re.sub(r"\s+", " ", re.sub(r"[,;:]+", " ", remainder)).strip(" -–—.")
        raag = {"printed": printed, "key": r["key"], "parent_key": r["parent"], "confidence": r["confidence"]}
    if taal:
        taal["laya"] = laya
        if taal["matras"] is None and printed_matras:
            taal["matras"] = printed_matras
    return {"raw": raw, "number": number, "raag": raag, "taal": taal, "laya": laya, "section": section,
            "reet_of": reet_of}


def _find_taal(text: str) -> dict | None:
    """The taal named in a heading, with the span of the words that name it."""
    toks = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]
    best = None
    for n in (3, 2, 1):
        for i in range(0, len(toks) - n + 1):
            s, e = toks[i][0], toks[i + n - 1][1]
            cand = text[s:e]
            got = normalise_taal(cand)
            # a one- or two-word window may also be a misread name (ਕਹਿਲਵਾ)
            if got["key"] and (got["method"] in ("alias", "skeleton") or (n <= 2 and got["method"] == "fuzzy")):
                if best is None or (e - s) > (best["span"][1] - best["span"][0]):
                    best = {"printed": " ".join(cand.strip(" ,()").split()), "key": got["key"], "matras": got["matras"],
                            "laya": None, "confidence": got["confidence"], "span": (s, e)}
        if best:
            return best
    # last resort: the whole text, fuzzily
    got = normalise_taal(text)
    if got["key"]:
        return {"printed": " ".join(text.split()), "key": got["key"], "matras": got["matras"], "laya": None,
                "confidence": got["confidence"], "span": (0, len(text))}
    return None


def _strip_section(text: str) -> str:
    from lib.notation_vocab import SECTIONS
    out = text
    for labels in SECTIONS.values():
        for label in labels:
            out = out.replace(label, " ")
    return re.sub("[%s]+" % DIGITS, " ", out)


# ---- cell tokens -----------------------------------------------------------

def swara_token(tok: str, script: str = "gurmukhi") -> dict | None:
    """
    {"swar": "S".."N", "glyph": tok, "khatka": bool} for a swara cell,
    {"held": True} for an extension mark, {"rest": True} for a rest, None
    for anything else. Brackets around a swara mean khatka: (ਮ).
    """
    t = (tok or "").strip()
    if not t:
        return None
    if t in _EXT_MARKS or set(t) <= set("—–―-=_"):
        return {"held": True}
    if t in SYMBOLS["rest"]:
        return {"rest": True}
    khatka = False
    if len(t) >= 3 and t[0] in "([" and t[-1] in ")]":
        t, khatka = t[1:-1], True
    core = t.strip(_PUNCT).replace("਼", "")          # a nukta is never a swara's mark
    s = swara_read(core, script)
    octave = None
    if s is None and script == "gurmukhi":
        # OCR confusions the fonts invite: a nukta, and the octave dots read
        # as vowel signs -- a dot below the letter comes back as ੁ or ੂ
        # (mandra), a dot above as ੰ or ਂ (taar). The letter is the swara;
        # the sign is a hint the grid reader weighs against the pixels.
        bare = core.replace("਼", "")
        stripped = _OCTAVE_SIGNS.sub("", bare)
        s = swara_read(stripped, script)
        if s is not None and stripped != bare:
            if _BELOW_SIGNS.search(bare):
                octave = -1
            elif _ABOVE_SIGNS.search(bare):
                octave = 1
    if s is None:
        return None
    out = {"swar": s, "glyph": tok, "khatka": khatka}
    if octave is not None:
        out["octave_hint"] = octave
    return out


def is_marker(tok: str) -> str | None:
    """'sam' | 'khali' | 'tali' when the token is a taal marker glyph."""
    return marker_kind((tok or "").strip())


def is_matra_number(tok: str) -> int | None:
    t = (tok or "").strip(_PUNCT)
    if t and _NUM.fullmatch(t):
        n = _int(t)
        return n if 1 <= n <= 32 else None
    return None


def clean_bol(tok: str) -> tuple[str | None, int]:
    """
    ("ਤਰਿਆ", 1) from "ਤਰਿਆऽ"; (None, 2) from "ऽऽ" or "SS" or "55" (Tesseract
    reads the avagraha as S or 5). A lone ਸ in a bol cell is a held mark too.
    """
    t = (tok or "").strip().strip(_PUNCT)
    if not t:
        return None, 0
    m = _HELD_TAIL.search(t)
    held = 0
    if m:
        tail = m.group(0)
        # a word ending in ਸ (ਦਾਸ, ਰਸ) is a word; only a bare ਸ or a run after
        # other letters that is all marks counts as held
        if tail == t or set(tail) & set("ऽ਽S5") or (tail == "ਸ" and len(t) == 1):
            held = len(tail)
            t = t[: -len(tail)]
        elif tail.startswith("ਸ"):
            pass
    if "-" in t and set(t) <= set("-—–"):
        return None, max(1, len(t))
    return (t or None), held


def bol_text(beats: list[dict]) -> str:
    """The syllables of a row joined for matching against the corpus: held marks, digits and punctuation dropped."""
    out = []
    for b in beats:
        bol = b.get("bol")
        if isinstance(bol, dict) and bol.get("g"):
            out.append(bol["g"])
    return " ".join(out)


def token_class(tok: str, script: str = "gurmukhi") -> str:
    """swara | marker | matra | bar | held | rest | gurmukhi | latin | other, for row-role voting."""
    t = (tok or "").strip()
    if not t:
        return "other"
    if t in SYMBOLS["bar"] or set(t) <= set("|[]!।"):
        return "bar"
    if _HELD_LIKE.match(t):
        return "held"
    st = swara_token(t, script)
    if st:
        if st.get("held"):
            return "held"
        if st.get("rest"):
            return "rest"
        return "swara"
    if is_marker(t):
        return "marker"
    if is_matra_number(t) is not None:
        return "matra"
    if re.search("[਀-੿]", t):
        return "gurmukhi"
    if re.search("[A-Za-z]", t):
        return "latin"
    return "other"
