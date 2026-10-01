"""
What a reader sees of a notation, derived from the record: never stored.

Four views, each a pure function of the record (lib/notation.py) and the
vocabulary (lib/notation_vocab.py):

  cells(rec)          the grid as rows of strings, per printed line: swaras
                      in Gurmukhi and in Latin, the bol in Gurmukhi and
                      roman, the taal markers, the matra numbers
  text_gurmukhi(rec)  one line of text per printed row, Gurmukhi swaras
  text_english(rec)   the same in Latin letters -- also the form a reviewer
                      edits in gold, so parse_line() inverts it exactly
  html(rec, ...)      the table the review page and the web app show

packages/search-core/src/notation-render.js is the same code in JavaScript;
fixtures/notations/*.expected.* pin both to the byte. Change a rule here,
regenerate the fixtures (test_notation.py --write-expected), and the
JavaScript test says whether the twin still agrees.

The Latin convention is the one the English keertan books already use:
shuddh S R G M P D N, komal in lowercase r g d n, tivra Ma as m; taar
saptak with a following apostrophe (S'), mandra with a comma (S,); a held
beat is -, a rest *, several notes in one beat are written together (DN),
a kan in braces before its note ({P}M), a khatka with a tilde (M~).
"""
from __future__ import annotations
import html as _html
import re

from lib.notation import fill_defaults
from lib.notation_vocab import ROMAN, SWARAS, TAALS, taal_markers

EXT, REST, UNKNOWN = "-", "*", "?"
EXT_PA, HELD_PA = "—", "ऽ"                   # — and ऽ
KOMAL_MARK, TIVRA_MARK, TAAR_MARK, MANDRA_MARK = "̲", "́", "̇", "̣"

NOTATION_CSS = """.ntn{font-family:inherit}.ntn-line{border-collapse:collapse;margin:.4em 0}
.ntn-cell{padding:.1em .45em;text-align:center;vertical-align:bottom;white-space:nowrap}
.ntn-vb{border-left:1px solid currentColor}.ntn-line tr:first-child .ntn-cell{padding-top:.6em}
.ntn-row-swar .ntn-cell{font-size:1.25em;line-height:1.1}.ntn-row-mark .ntn-cell{font-size:.85em;opacity:.75}
.ntn-row-m .ntn-cell{font-size:.7em;opacity:.55;font-weight:normal}
.n{position:relative;display:inline-block;padding:0 .05em}
.n-komal{text-decoration:underline;text-underline-offset:.15em}
.n-tivra::after{content:"|";position:absolute;top:-.55em;left:50%;transform:translateX(-50%);font-size:.6em;line-height:1}
.n-taar::before,.n-mandra::after{content:"\\2022";position:absolute;left:50%;transform:translateX(-50%);font-size:.45em;line-height:1}
.n-taar::before{top:-.6em}.n-mandra::after{bottom:-.55em}
.n-taar2::before{content:"\\2022\\2022"}.n-mandra2::after{content:"\\2022\\2022"}
.n-grp{display:inline-block;border-bottom:1px solid currentColor;border-radius:0 0 .5em .5em;padding:0 .1em}
.n-kan{font-size:.6em;vertical-align:super;margin-right:-.1em}.n-khatka::after{content:"~";font-size:.6em;vertical-align:super}
.n-rest,.n-ext{opacity:.7}.n-unknown{color:#b00;border-bottom:1px dotted #b00}
.ntn-unknown{background:rgba(255,0,0,.06)}.b-held{opacity:.6}
.ntn-sam{font-weight:600}.ntn-khali .ntn-cell,.ntn-row-mark .ntn-khali{opacity:.9}
.ntn-sec-label{font-weight:600;margin-top:.8em}.ntn-cont{border-left-style:dashed}
"""


# ---- one note, one beat ----------------------------------------------------

def note_latin(note: dict) -> str:
    """{'s':'R','k':True,'o':1} -> "r'"; a kan is prefixed in braces, a khatka suffixed with ~."""
    s = note["s"]
    letter = s
    if note.get("k") and s in SWARAS["latin_komal"]:
        letter = SWARAS["latin_komal"][s]
    elif note.get("t") and s in SWARAS["latin_tivra"]:
        letter = SWARAS["latin_tivra"][s]
    out = letter + _octave_suffix(note.get("o", 0))
    if note.get("kan"):
        out = "{" + note_latin({**note["kan"], "kan": None, "kh": False}) + "}" + out
    if note.get("kh"):
        out += "~"
    return out


def note_gurmukhi(note: dict, kanna: bool = False) -> str:
    """The same note in Gurmukhi letters with combining marks: komal ਰ̲, tivra ਮ́, taar ਸ̇, mandra ਸ̣."""
    s = note["s"]
    letter = (SWARAS["gurmukhi_kanna"] if kanna else SWARAS["gurmukhi"])[s]
    if note.get("k"):
        letter += KOMAL_MARK
    if note.get("t"):
        letter += TIVRA_MARK
    o = note.get("o", 0)
    letter += TAAR_MARK * o if o > 0 else MANDRA_MARK * -o
    if note.get("kan"):
        letter = "{" + note_gurmukhi({**note["kan"], "kan": None, "kh": False}, kanna) + "}" + letter
    if note.get("kh"):
        letter += "~"
    return letter


def _octave_suffix(o: int) -> str:
    return "'" * o if o > 0 else "," * -o


def beat_cell(beat: dict, script: str = "english", kanna: bool = False) -> str:
    """The swar cell of one beat: notes run together, a note's extra length as trailing dashes."""
    if beat.get("ext"):
        return EXT if script == "english" else EXT_PA
    if beat.get("rest"):
        return REST
    notes = beat.get("notes")
    if notes is None:
        return UNKNOWN
    parts = []
    for n in notes:
        txt = note_latin(n) if script == "english" else note_gurmukhi(n, kanna)
        parts.append(txt + EXT * (n.get("len", 1) - 1))
    return "".join(parts)


def bol_cell(beat: dict, script: str = "english", roman: str | None = None) -> str:
    """The bol cell: the syllable and its held marks (ऽ in Gurmukhi, - in roman)."""
    bol = beat.get("bol")
    if bol is None:
        return ""
    if script == "english":
        base = roman if roman is not None else roman_of(bol.get("g", ""))
        return base + EXT * bol.get("h", 0)
    return bol.get("g", "") + HELD_PA * bol.get("h", 0)


# ---- roman bol -------------------------------------------------------------

def roman_of(g: str) -> str:
    """
    A plain roman rendering of a Gurmukhi syllable from the vocabulary's
    table: a consonant cluster takes its vowel sign, or the inherent 'a'
    unless the word ends there; a word-final ੁ or ਿ is silent (ਸਤਿ -> sat);
    an addak doubles the consonant after it (ਸੱਚ -> sacch). A stand-in for
    the corpus's own transliteration, used where a beat's syllable is only
    part of a word.
    """
    cons, signs, ind, nasal, sub = (ROMAN["consonants"], ROMAN["vowel_signs"], ROMAN["independent"],
                                    ROMAN["nasal"], ROMAN["subjoined"])
    inherent = ROMAN["inherent"]
    chars: list[str] = []
    for ch in g or "":
        if ch == "\u0a3c" and chars and chars[-1] + ch in cons:       # nukta letter as two code points
            chars[-1] += ch
        else:
            chars.append(ch)
    out: list[str] = []
    i, n = 0, len(chars)
    while i < n:
        ch = chars[i]
        if ch in cons:
            out.append(cons[ch])
            i += 1
            while i + 1 < n and chars[i] == "\u0a4d" and chars[i + 1] in cons:     # subjoined letter
                out.append(sub.get(chars[i] + chars[i + 1], cons[chars[i + 1]]))
                i += 2
            if i < n and chars[i] in signs:
                if not (chars[i] in ("\u0a41", "\u0a3f") and i == n - 1):
                    out.append(signs[chars[i]])
                i += 1
            elif i < n and chars[i] == "\u0a71":
                out.append(inherent)
                if i + 1 < n and chars[i + 1] in cons:
                    out.append(cons[chars[i + 1]][0])
                i += 1
            elif i < n:
                out.append(inherent)
            continue
        if ch in ind:
            out.append(ind[ch])
        elif ch in nasal:
            out.append(nasal[ch])
        elif ch in signs:
            out.append(signs[ch])
        i += 1
    return "".join(out)


def word_spans(text: str) -> list[tuple[int, int]]:
    """[start, end) of every word of a corpus line, skipping the segments clean_gurmukhi drops (counters, rahao markers)."""
    from lib.gurmukhi_text import _DIGITS, _MARKER
    spans = []
    for seg in re.finditer("[^।॥]+", text):
        body = seg.group(0)
        if not body.strip() or _DIGITS.match(body) or _MARKER.match(body):
            continue
        for m in re.finditer(r"\S+", body):
            spans.append((seg.start() + m.start(), seg.start() + m.end()))
    return spans


def roman_for_line(line: dict, corpus_line: dict | None) -> list[str | None]:
    """
    The roman bol of each beat: the corpus's own transliteration where a
    beat's span is a whole word of the line, else None (the caller falls
    back to roman_of).
    """
    beats = line.get("beats") or []
    if not corpus_line or not corpus_line.get("gurmukhi_uni") or not corpus_line.get("translit_roman"):
        return [None] * len(beats)
    words = word_spans(corpus_line["gurmukhi_uni"])
    roman = corpus_line["translit_roman"].split()
    if len(words) != len(roman):
        return [None] * len(beats)
    by_span = {span: roman[i] for i, span in enumerate(words)}
    out: list[str | None] = []
    for beat in beats:
        bol = beat.get("bol")
        span = tuple(bol["span"]) if isinstance(bol, dict) and bol.get("span") else None
        out.append(by_span.get(span))
    return out


# ---- the grid --------------------------------------------------------------

def cells(rec: dict, corpus_lines: dict | None = None, kanna: bool = False) -> list[dict]:
    """
    Per printed line: {"section", "n", "label", "kind", "avartan", "matra_from", "taal",
    "continues", "beats", "matras", "marks", "vibhag", "swar_pa", "swar_en", "bol_pa",
    "bol_en", "unknown", "has_bol"}, every per-beat list the same length.
    `corpus_lines` maps line_id -> {"gurmukhi_uni", "translit_roman"}.
    """
    rec = fill_defaults(rec)
    corpus_lines = corpus_lines or {}
    top_taal = (rec.get("heading") or {}).get("taal")
    out = []
    for sec in rec.get("sections") or []:
        own = sec.get("taal")                            # a section's taal is its key; the heading's is an object
        taal = ({"key": own} if isinstance(own, str) else own) or top_taal
        key = (taal or {}).get("key")
        info = TAALS.get(key) if key else None
        vibhag_starts = _vibhag_starts(info)
        for line in sec.get("lines") or []:
            beats = line.get("beats") or []
            n = len(beats)
            start = line.get("matra_from", 1)
            free = line.get("kind") == "free"
            marks = [None] * n if free or not info else taal_markers(key, start, n)
            lid = line.get("line_id")
            corpus_line = None if lid is None else (corpus_lines.get(lid) or corpus_lines.get(str(lid)))
            roman = roman_for_line(line, corpus_line)
            out.append({
                "section": sec.get("kind"), "n": sec.get("n"), "label": sec.get("label"),
                "kind": line.get("kind"), "avartan": line.get("avartan"), "matra_from": start, "taal": key,
                "continues": bool(line.get("continues")),
                "beats": beats,
                "matras": [None if free else b.get("m") for b in beats],
                "marks": marks,
                "vibhag": [False if free else ((b.get("m") or 0) in vibhag_starts) for b in beats],
                "swar_pa": [beat_cell(b, "gurmukhi", kanna) for b in beats],
                "swar_en": [beat_cell(b, "english") for b in beats],
                "bol_pa": [bol_cell(b, "gurmukhi") for b in beats],
                "bol_en": [bol_cell(b, "english", roman[i]) for i, b in enumerate(beats)],
                "unknown": ["notes" in b and b["notes"] is None for b in beats],
                "has_bol": any("bol" in b for b in beats),
            })
    return out


def _vibhag_starts(info: dict | None) -> set[int]:
    if not info or not info.get("vibhag"):
        return set()
    starts, m = set(), 1
    for size in info["vibhag"]:
        starts.add(m)
        m += size
    return starts


def format_line(row: dict, script: str = "english") -> str:
    """One printed row as text: beats separated by spaces, '|' before each vibhag, '||' closing an avartan."""
    swar = row["swar_en" if script == "english" else "swar_pa"]
    parts = []
    for i, cell in enumerate(swar):
        if row["vibhag"][i] and i > 0:
            parts.append("|")
        parts.append(cell)
    line = " ".join(parts)
    if row["kind"] == "avartan":
        if row["matra_from"] != 1:
            line = "@%d %s" % (row["matra_from"], line)
        line += " ||"
    return line


def text_english(rec: dict, corpus_lines: dict | None = None) -> str:
    return _text(rec, "english", corpus_lines)


def text_gurmukhi(rec: dict, corpus_lines: dict | None = None) -> str:
    return _text(rec, "gurmukhi", corpus_lines)


def _text(rec: dict, script: str, corpus_lines: dict | None) -> str:
    lines = []
    last_sec = None
    for row in cells(rec, corpus_lines):
        sec_key = (row["section"], row["n"])
        if sec_key != last_sec:
            lines.append("[%s%s]" % (row["section"], " %d" % row["n"] if row["n"] else ""))
            last_sec = sec_key
        lines.append(format_line(row, script))
        if row["has_bol"]:
            bol = row["bol_en" if script == "english" else "bol_pa"]
            lines.append("  " + " ".join(b if b else "." for b in bol))
    return "\n".join(lines)


# ---- the text form, inverted (gold editing) --------------------------------

_TOKEN = re.compile(r"(\{[^}]*\})?([SRGMPDNsrgmpdn])([',]*)(~?)")


def parse_line(text: str) -> dict:
    """
    The inverse of format_line for the English form: '@9 S r- | DN * ||' ->
    {"kind": "avartan", "matra_from": 9, "beats": [...]}. Vibhag bars are
    display, so they are dropped; the taal supplies them again.
    """
    s = text.strip()
    kind = "avartan" if s.endswith("||") else "free"
    s = s[:-2].strip() if kind == "avartan" else s
    matra_from = 1
    m = re.match(r"^@(\d+)\s+", s)
    if m:
        matra_from = int(m.group(1))
        s = s[m.end():]
    beats = [parse_cell(tok) for tok in s.split() if tok != "|"]
    return {"kind": kind, "matra_from": matra_from, "beats": beats}


def parse_cell(tok: str) -> dict:
    """'{P}m~-' -> one beat: a tivra Ma with a kan of Pa, two units long in a beat divided in two."""
    if tok == EXT:
        return {"ext": True}
    if tok == REST:
        return {"rest": True}
    if tok == UNKNOWN:
        return {"notes": None}
    notes: list[dict] = []
    i = 0
    while i < len(tok):
        m = _TOKEN.match(tok, i)
        if not m:
            raise ValueError("cannot read swara cell %r at %d" % (tok, i))
        kan, letter, oct_, kh = m.groups()
        note = _note_from_letter(letter)
        o = oct_.count("'") - oct_.count(",")
        if o:
            note["o"] = o
        if kan:
            inner = kan[1:-1]
            note["kan"] = _note_from_letter(inner[0])
            ko = inner.count("'") - inner.count(",")
            if ko:
                note["kan"]["o"] = ko
        if kh:
            note["kh"] = True
        i = m.end()
        ln = 1
        while i < len(tok) and tok[i] == EXT:
            ln += 1
            i += 1
        if ln > 1:
            note["len"] = ln
        notes.append(note)
    total = sum(n.get("len", 1) for n in notes)
    beat: dict = {"notes": notes}
    if total != 1:
        beat["div"] = total
    return beat


def _note_from_letter(ch: str) -> dict:
    up = ch.upper()
    note = {"s": up}
    if ch.islower():
        if up == "M":
            note["t"] = True
        else:
            note["k"] = True
    return note


# ---- HTML ------------------------------------------------------------------

def html(rec: dict, script: str = "gurmukhi", corpus_lines: dict | None = None, matra_row: bool = True,
         kanna: bool = False) -> str:
    """
    <div class="ntn"> with one <table class="ntn-line"> per printed row; the
    same DOM whichever script, only the text differs. Everything is escaped.
    """
    e = _html.escape
    rows = cells(rec, corpus_lines, kanna)
    out = ['<div class="ntn" data-id="%s" data-script="%s">' % (e(str(rec.get("notation_id", ""))), e(script))]
    last_sec = None
    for row in rows:
        sec_key = (row["section"], row["n"])
        if sec_key != last_sec:
            if last_sec is not None:
                out.append("</div>")
            label = row["label"] or row["section"] or ""
            out.append('<div class="ntn-sec" data-kind="%s" data-n="%s"><div class="ntn-sec-label">%s</div>'
                       % (e(str(row["section"])), e(str(row["n"] or "")), e(label)))
            last_sec = sec_key
        free = row["kind"] == "free"
        cls = "ntn-line" + (" ntn-free" if free else "") + (" ntn-cont" if row["continues"] else "")
        out.append('<table class="%s" data-kind="%s" data-taal="%s" data-from="%d">'
                   % (cls, e(row["kind"]), e(row["taal"] or ""), row["matra_from"]))
        n = len(row["swar_en"])
        cell_cls = []
        for i in range(n):
            c = ["ntn-cell"]
            if row["vibhag"][i]:
                c.append("ntn-vb")
            mark = row["marks"][i]
            if mark is not None:
                c.append({"×": "ntn-sam", "0": "ntn-khali"}.get(mark, "ntn-tali"))
            if row["unknown"][i]:
                c.append("ntn-unknown")
            cell_cls.append(" ".join(c))
        m_txt = ["" if row["matras"][i] is None else str(row["matras"][i]) for i in range(n)]
        if matra_row and not free:
            out.append('<tr class="ntn-row ntn-row-m">' + "".join(
                '<th class="%s">%s</th>' % (cell_cls[i], m_txt[i]) for i in range(n)) + "</tr>")
        swar_cells = _swar_html(row, script, kanna)
        out.append('<tr class="ntn-row ntn-row-swar">' + "".join(
            '<td class="%s" data-m="%s">%s</td>' % (cell_cls[i], m_txt[i], swar_cells[i]) for i in range(n)) + "</tr>")
        if row["has_bol"]:
            bol = row["bol_en" if script == "english" else "bol_pa"]
            out.append('<tr class="ntn-row ntn-row-bol">' + "".join(
                '<td class="%s">%s</td>' % (cell_cls[i], _bol_html(bol[i], script)) for i in range(n)) + "</tr>")
        if not free and any(mk is not None for mk in row["marks"]):
            out.append('<tr class="ntn-row ntn-row-mark">' + "".join(
                '<td class="%s">%s</td>' % (cell_cls[i], e(row["marks"][i] or "")) for i in range(n)) + "</tr>")
        out.append("</table>")
    if last_sec is not None:
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


def _swar_html(row: dict, script: str, kanna: bool) -> list[str]:
    """The swar cells of one row as markup: each note a <span class="n ...">, a group wrapped in n-grp."""
    e = _html.escape
    out = []
    for beat in row["beats"]:
        if beat.get("ext"):
            out.append('<span class="n n-ext">%s</span>' % (EXT if script == "english" else EXT_PA))
            continue
        if beat.get("rest"):
            out.append('<span class="n n-rest">%s</span>' % REST)
            continue
        notes = beat.get("notes")
        if notes is None:
            out.append('<span class="n n-unknown" title="%s">%s</span>' % (e(str(beat.get("raw", ""))), UNKNOWN))
            continue
        inner = "".join(_note_html(note, script, kanna) for note in notes)
        if len(notes) > 1:
            inner = '<span class="n-grp n-grp-%d">%s</span>' % (min(len(notes), 4), inner)
        out.append(inner)
    return out


def _note_html(note: dict, script: str, kanna: bool) -> str:
    e = _html.escape
    classes = ["n"]
    if note.get("k"):
        classes.append("n-komal")
    if note.get("t"):
        classes.append("n-tivra")
    o = note.get("o", 0)
    if o >= 2:
        classes.append("n-taar2")
    elif o == 1:
        classes.append("n-taar")
    elif o == -1:
        classes.append("n-mandra")
    elif o <= -2:
        classes.append("n-mandra2")
    if note.get("kh"):
        classes.append("n-khatka")
    ln = note.get("len", 1)
    if ln > 1:
        classes.append("n-len-%d" % ln)
    s = note["s"]
    if script == "english":
        if note.get("k"):
            letter = SWARAS["latin_komal"].get(s, s)
        elif note.get("t"):
            letter = SWARAS["latin_tivra"].get(s, s)
        else:
            letter = s
        letter += _octave_suffix(o)
    else:
        letter = (SWARAS["gurmukhi_kanna"] if kanna else SWARAS["gurmukhi"])[s]
    body = '<span class="%s" data-s="%s">%s</span>' % (" ".join(classes), s, e(letter))
    if note.get("kan"):
        kan = note["kan"]
        kl = note_latin({**kan, "kan": None, "kh": False}) if script == "english" else SWARAS["gurmukhi"][kan["s"]]
        body = '<sup class="n-kan">%s</sup>' % e(kl) + body
    return body + (EXT * (ln - 1) if ln > 1 else "")


def _bol_html(cell: str, script: str) -> str:
    e = _html.escape
    held = HELD_PA if script != "english" else EXT
    core = cell.rstrip(held)
    h = len(cell) - len(core)
    out = e(core)
    if h:
        out += '<span class="b-held">%s</span>' % e(held * h)
    return out
