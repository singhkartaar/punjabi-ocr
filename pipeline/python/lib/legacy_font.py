"""
Text typed in a legacy Gurmukhi font, as Unicode.

Before Unicode, Gurmukhi was typed in fonts that drew its letters over the
Latin keyboard: GurbaniAkhar, AnmolLipi, GurbaniLipi and their kin share one
layout, in which "suxI pukwr dwqwr" is what the keys of ਸੁਣੀ ਪੁਕਾਰ ਦਾਤਾਰ
are. A PDF set in such a font has a real text layer, and the text layer is
those keystrokes. Turning them into Unicode is a table and a few rules about
order, not a reading: the sihari is typed BEFORE the consonant it follows in
Unicode ("ik" is ਕਿ), some vowels are two keys ("Aw" is ਆ), and a mark typed
before the one it should follow is put back ("uN" and "Nu" are both ੁਂ).

This is the ASCII-to-Unicode half of anvaad-js 1.5.1 (KhalisFoundation, MIT;
src/unicode.js), in Python so that a book in a legacy font needs nothing but
this repository. The table is that file's, key for key, and test_ocr.py
holds examples from that library's own tests with what it returns for them.

One deliberate difference. Where a sihari is followed by a character the
table does not have ("i," or "i" before a letter already in Unicode),
anvaad-js writes the word "undefined" into the text; here the character is
kept as it is. Over that library's 52 examples and 300,000 random strings,
that is the only place the two disagree (972 of the random ones).
"""
from __future__ import annotations

# the keyboard: what each key draws
MAPPING = {
    "0": "੦", "1": "੧", "2": "੨", "3": "੩", "4": "੪", "5": "੫", "6": "੬", "7": "੭", "8": "੮",
    "9": "੯", "a": "ੳ", "A": "ਅ", "s": "ਸ", "S": "ਸ਼", "d": "ਦ", "D": "ਧ", "f": "ਡ", "F": "ਢ",
    "g": "ਗ", "G": "ਘ", "h": "ਹ", "H": "੍ਹ", "j": "ਜ", "J": "ਝ", "k": "ਕ", "K": "ਖ", "l": "ਲ",
    "L": "ਲ਼", "q": "ਤ", "Q": "ਥ", "w": "ਾ", "W": "ਾਂ", "e": "ੲ", "E": "ਓ", "r": "ਰ", "R": "੍ਰ",
    "®": "੍ਰ", "t": "ਟ", "T": "ਠ", "y": "ੇ", "Y": "ੈ", "u": "ੁ", "ü": "ੁ", "U": "ੂ", "¨": "ੂ",
    "i": "ਿ", "I": "ੀ", "o": "ੋ", "O": "ੌ", "p": "ਪ", "P": "ਫ", "z": "ਜ਼", "Z": "ਗ਼", "x": "ਣ",
    "X": "ਯ", "c": "ਚ", "C": "ਛ", "v": "ਵ", "V": "ੜ", "b": "ਬ", "B": "ਭ", "n": "ਨ", "ƒ": "ਨੂੰ",
    "N": "ਂ", "ˆ": "ਂ", "m": "ਮ", "M": "ੰ", "µ": "ੰ", "`": "ੱ", "~": "ੱ", "¤": "ੱ", "Í": "੍ਵ",
    "ç": "੍ਚ", "†": "੍ਟ", "œ": "੍ਤ", "˜": "੍ਨ", "´": "ੵ", "Ï": "ੵ", "æ": "਼", "Î": "੍ਯ",
    "ì": "ਯ", "í": "੍ਯ", "^": "ਖ਼", "&": "ਫ਼", "\\": "ਞ", "|": "ਙ", "[": "।", "]": "॥", "<": "ੴ",
    "¡": "ੴ", "Å": "ੴ", "Ú": "ਃ", "Ç": "☬", "@": "ੑ", "‚": "❁", "•": "੶", " ": " ",
}

# A mark typed before the mark it belongs after. Each entry is the right
# order; its reverse is what gets typed, and is turned round before anything
# else is read.
CORRECTIONS = ("@W", "@w", "@o", "@O", "@y", "@Y", "@ü", "@`", "ÍY", "Ry", "RY", "RM", "RN", "YN", "yN",
               "YM", "yM", "uN", "UN", "üN", "uM", "UM", "üM", "R`", "u`", "U`", "ü`", "Iˆ", "IN")

# What is written under or after a consonant and so stays with it when a
# sihari is moved: "ikR" is ਕ + ੍ਰ + ਿ, not ਕ + ਿ + ੍ਰ.
HALF = frozenset("HR®Íç†œ˜´ÎÏíæ")

# Keys that draw nothing of their own: the second half of ੴ typed as "<>",
# and two spacing characters the fonts use to place a mark.
DROPPED = str.maketrans("", "", ">\u00d8\u00f8\u00c6")

SIHARI = "ਿ"
VOWELS = {("a", "u"): "ਉ", ("a", "U"): "ਊ",
          ("A", "w"): "ਆ", ("A", "W"): "ਆਂ", ("A", "Y"): "ਐ", ("A", "O"): "ਔ",
          ("e", "I"): "ਈ", ("e", "y"): "ਏ",
          ("u", "o"): "ੋੁ"}


def to_unicode(text: str) -> str:
    """
    @param text  one line as the legacy font's keystrokes; text already in
                 Unicode passes through, since no Gurmukhi letter is a key
    @returns     the same line in Unicode Gurmukhi
    """
    if not text:
        return text
    s = text.translate(DROPPED)
    for right in CORRECTIONS:
        s = s.replace(right[::-1], right)

    out: list[str] = []
    j, n = 0, len(s)
    while j < n:
        c = s[j]
        nxt = s[j + 1] if j + 1 < n else None
        after = s[j + 2] if j + 2 < n else None
        if c == "i" and nxt is not None:
            if nxt == "e":
                out.append("ਇ")
            elif after in HALF:
                out += [MAPPING.get(nxt, nxt), MAPPING[after], SIHARI]
                j += 1
            else:
                out += [MAPPING.get(nxt, nxt), SIHARI]
            j += 2
        elif (c, nxt) in VOWELS:
            out.append(VOWELS[(c, nxt)])
            j += 2
        elif c == "1" and nxt == "E" and after == "\u00e5":
            out.append("ੴ")
            j += 3
        else:
            out.append(MAPPING.get(c, c))
            j += 1
    return "".join(out)
