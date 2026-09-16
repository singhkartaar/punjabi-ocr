"""
Finding the Gurbani a passage of English prose is quoting.

The essays quote Gurbani in English translation and then give the ang it came
from. Resolving that to a shabad id is what lets an answer drawn from the prose
show the verse it rests on, and open it in full.

Two stages, kept separate because conflating them is why this kind of thing
usually cannot explain its false positives:

  detect   which span of the page is a quotation. On the two-up spreads this is
           read off the page geometry -- quotations are set in italic -- and
           lib/writings_pdf.py has already marked them. Elsewhere a quotation is
           found from its own punctuation: a trailing ang, "[Pg. 289]", or the
           "|| 1 ||" stanza counters the translations carry.

  resolve  which shabad it is. Layered, each layer running only where the one
           before it did not fire, and AMBIGUOUS IS NEVER GUESSED -- a wrong
           link sends a reader to the wrong verse and is invisible, while an
           unresolved one is visibly incomplete. The same discipline is written
           into lib/mt-rekey.js and alignDarpan.

    L1  the ang. On its own this resolves nothing -- an ang holds four to eight
        shabads -- it is the window the others run inside.
    L2  the English of the quote against every line on that ang, in all three
        translations the corpus holds. Bau Ji quotes the published English close
        to verbatim, so this carries the work: measured at 88.5% on a 12-file
        prototype, most of them at a containment of 1.00.
    L3  (13_resolve_citations.py) a vector fallback for the rest, inside the
        same ang window.

Three translations are scored, not one, because he uses more than one: probes
matched bdb/ssk on angs 478, 921 and 791 and ms on 818 and 1417. `ssk` is no
longer a shipped index but is still in corpus.sqlite, and this is a build-time
join against that file, so its absence from artifacts/ costs nothing.
"""
from __future__ import annotations
import re

# The ang as the essays write it. "[Pg. 289]" and "(661)" are explicit; far more
# often it is simply the number left at the end of the quote, sometimes as a
# range across two angs ("1349-1350"), sometimes with the author named after it
# ("1349-1350 Bh. Kabir"), and sometimes glued to the last word by the PDF
# ("emotional attachment and love of duality well up.921").
PG = re.compile(r"\[\s*(?:Pg|Page|Ang|P)\.?\s*([0-9]{1,4})\s*(?:-\s*[0-9]{1,4}\s*)?\]", re.I)
PAREN = re.compile(r"\(\s*([0-9]{2,4})\s*(?:-\s*[0-9]{2,4}\s*)?\)\s*[^0-9]{0,24}$")
TRAILING = re.compile(r"(?:^|[^0-9])([0-9]{2,4})\s*(?:-\s*[0-9]{2,4})?\s*[^0-9]{0,24}$")
# the stanza counters the English translations carry, and the rahao marker
COUNTER = re.compile(r"\|\|\s*[0-9]*\s*\|\||\|\|\s*Pause\s*\|\||\bPause\b\s*\|\||\.\s*Pause\s*\.", re.I)
MIN_ANG, MAX_ANG = 2, 1430

# Words that carry no signal about which verse this is. Kept deliberately short:
# the scoring is containment against a corpus line, so a long stop list would
# strip the very words that make a short line distinctive.
STOP = set("""the a an of is are was were to in and or for with by that this his her its it
he she they we you i as be been on at from not but so all no nor own do does did have has
had will would shall should may might can could am me my your their our them us who whom
which what when where why how than then there here such very just only also even if""".split())
WORD = re.compile(r"[a-z0-9]+")
MIN_TOKENS = 5          # a quote with fewer content words cannot be told apart
ACCEPT = 0.45           # containment of a corpus line's words by the quote
MARGIN = 0.10           # by which the best shabad must beat the next
# Containment alone makes SHORT corpus lines into score magnets: a line of three
# content words scores a perfect 1.00 against any long quotation that happens to
# use all three, so the same few shabads came back for unrelated quotes. The
# overlap must also be substantial in absolute terms.
MIN_OVERLAP = 4
ANG_SLACK = 1           # a cited ang can be one out where a shabad spans a page break


def tokens(text: str) -> set[str]:
    """Content words, lower case. Order is not kept: the quote may be edited."""
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def find_ang(text: str):
    """
    The ang a quotation cites, and the quotation with the citation removed.

    @returns (ang | None, text without the citation)
    """
    m = PG.search(text)
    if m and MIN_ANG <= int(m.group(1)) <= MAX_ANG:
        return int(m.group(1)), PG.sub(" ", text).strip()
    for rx in (PAREN, TRAILING):
        m = rx.search(text)
        if not m:
            continue
        ang = int(m.group(1))
        if MIN_ANG <= ang <= MAX_ANG:
            return ang, (text[:m.start(1)] + " " + text[m.end(1):]).strip()
    return None, text


def clean_quote(text: str) -> str:
    """A quotation without its counters, its leading number, or its ang."""
    _, text = find_ang(text)
    text = COUNTER.sub(" ", text)
    text = re.sub(r"^\s*[0-9]{1,3}\s*[.)]?\s+", "", text)   # the essay's own numbering
    return re.sub(r"\s+", " ", text).strip(" .-–—")


def looks_quoted(text: str) -> bool:
    """
    True where a paragraph announces itself as a quotation without italics.

    Used for the books and the two-column essays, whose PDFs carry no slant. A
    trailing ang alone is not enough -- a page number would qualify -- so the
    number must be above the highest page number in this corpus, or be written
    as an explicit citation, or the line must carry the stanza counters.
    """
    if PG.search(text) or COUNTER.search(text):
        return True
    ang, _ = find_ang(text)
    return ang is not None and ang > 61


def detect(records: list[dict]) -> list[dict]:
    """
    The quotations among a work's paragraphs.

    @param records paragraph records from 12_ingest_writings.py
    @returns [{unit_id, work, part, page, para_no, ang, text, how}]
    """
    out = []
    for rec in records:
        italic = rec["style"] == "quote"
        if not italic and not looks_quoted(rec["text"]):
            continue
        ang, _ = find_ang(rec["text"])
        text = clean_quote(rec["text"])
        if len(tokens(text)) < MIN_TOKENS:
            continue
        out.append({"unit_id": rec["unit_id"], "work": rec["work"], "part": rec["part"],
                    "page": rec["page"], "para_no": rec["para_no"], "ang": ang,
                    "text": text, "how": "italic" if italic else "marked"})
    return out


def corpus_by_ang(con, translators=("ssk", "bdb", "ms")) -> dict:
    """
    {ang: [(shabad_id, line_id, tokens)]} over every English translation held.

    Read from corpus.sqlite, not from the shipped database: `ssk` was dropped
    from what ships but is still the closest match for a good share of these
    quotations.
    """
    marks = ",".join("?" * len(translators))
    rows = con.execute(
        "SELECT l.ang, l.shabad_id, l.line_id, t.text FROM lines l "
        "JOIN translations t ON t.line_id = l.line_id "
        "WHERE t.translator IN (%s) AND l.kind IN ('line','rahao')" % marks, translators)
    by_ang: dict[int, list] = {}
    for ang, shabad_id, line_id, text in rows:
        tok = tokens(text)
        if tok:
            by_ang.setdefault(ang, []).append((shabad_id, line_id, tok))
    return by_ang


def resolve_lexical(span: dict, by_ang: dict) -> dict | None:
    """
    Which shabad on the cited ang this quotation is, by word containment.

    Scores how much of a corpus LINE the quotation contains, not the reverse: a
    quotation often runs over several lines, so measuring against the quote
    would punish exactly the longest and most certain matches.

    @returns {shabad_id, line_id, score, margin, method} or None where the ang
             is missing, nothing scores, or two shabads score alike.
    """
    ang = span.get("ang")
    if not ang:
        return None
    q = tokens(span["text"])
    if len(q) < MIN_TOKENS:
        return None
    best: dict[int, tuple[float, int]] = {}
    for a in range(ang - ANG_SLACK, ang + ANG_SLACK + 1):
        for shabad_id, line_id, tok in by_ang.get(a, ()):
            shared = len(q & tok)
            if shared < MIN_OVERLAP:
                continue
            score = shared / float(len(tok))
            if score > best.get(shabad_id, (0.0, 0))[0]:
                best[shabad_id] = (score, line_id)
    if not best:
        return None
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    (shabad_id, (score, line_id)) = ranked[0]
    second = ranked[1][1][0] if len(ranked) > 1 else 0.0
    if score < ACCEPT:
        return None
    if score - second < MARGIN:
        return {"ambiguous": True, "score": round(score, 3), "runner_up": round(second, 3)}
    return {"shabad_id": shabad_id, "line_id": line_id, "score": round(score, 3),
            "margin": round(score - second, 3), "method": "lexical_ang"}


def precomputed(records: list[dict]) -> list[dict]:
    """
    Quotations an OCR reader already resolved against the corpus.

    lib/ocr_match.py found these lines in data/corpus.sqlite while the page
    was being merged (22_ocr_merge.py), from the Gurmukhi itself rather than
    from an English rendering, and lib/writings_ocr.py put the line_ids on the
    paragraph. They are taken as given here, method "ocr-corpus-match", and
    detect() never sees them: they are Gurmukhi and would fail MIN_TOKENS anyway.
    """
    out = []
    for rec in records:
        ids = rec.get("line_ids")
        if not ids:
            continue
        out.append({"unit_id": rec["unit_id"], "work": rec["work"], "part": rec["part"],
                    "page": rec["page"], "para_no": rec["para_no"], "ang": rec.get("ang"),
                    "text": rec["text"], "how": "ocr-corpus-match",
                    "shabad_id": rec.get("shabad_id"), "line_id": ids[0], "line_ids": list(ids),
                    "score": rec.get("match_score"), "method": "ocr-corpus-match"})
    return out
