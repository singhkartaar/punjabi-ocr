"""
What text an index embeds.

A source is `gurmukhi_uni` -- the scripture itself, cleaned of its counters
and markers -- or a translator id from the corpus (`bdb`, `pa-ss`, `en-ss-mt`
...), or a comma list of translator ids whose unit vectors are averaged per
line (the English index is `bdb,ms`). One loader serves the embedding step,
the document builder and the reference export, so no script decides on its
own what a source means.
"""
from __future__ import annotations
import sqlite3
from lib.gurmukhi_text import clean_gurmukhi

GURMUKHI_SOURCE = "gurmukhi_uni"


def parse_source(source: str) -> list[str]:
    parts = [s.strip() for s in str(source).split(",") if s.strip()]
    if not parts:
        raise ValueError("empty source")
    return parts


def source_lang(source: str) -> str:
    """'pa' for the scripture and the Punjabi teekas, 'en' for everything else."""
    parts = parse_source(source)
    langs = {("pa" if p == GURMUKHI_SOURCE or p.startswith("pa-") else "en") for p in parts}
    if len(langs) > 1:
        raise ValueError(f"a source must be one language, got {sorted(langs)} from {source!r}")
    return langs.pop()


def load_texts(con: sqlite3.Connection, source: str) -> dict[str, dict[int, str]]:
    """
    {source_id: {line_id: text}} for each part of the source. Gurmukhi lines
    are cleaned; a line that is only a marker keeps its published text so it
    stays embeddable. Translator texts are whitespace-normalised. A line a
    translator skipped is simply absent from its dict.
    """
    out = {}
    for part in parse_source(source):
        if part == GURMUKHI_SOURCE:
            texts = {}
            for lid, uni in con.execute("SELECT line_id, gurmukhi_uni FROM lines ORDER BY line_id"):
                cleaned = clean_gurmukhi(uni)
                texts[lid] = cleaned if cleaned else " ".join(uni.split())
            out[part] = texts
        else:
            rows = con.execute("SELECT line_id, text FROM translations WHERE translator=? ORDER BY line_id", (part,)).fetchall()
            if not rows:
                known = [r[0] for r in con.execute("SELECT DISTINCT translator FROM translations ORDER BY 1")]
                raise ValueError(f"translator {part!r} has no rows in the corpus; known: {known}")
            out[part] = {lid: " ".join(text.split()) for lid, text in rows if text and text.strip()}
    return out
