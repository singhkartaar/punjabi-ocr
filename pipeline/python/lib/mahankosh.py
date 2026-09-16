"""
Mahan Kosh (ਗੁਰ ਸ਼ਬਦ ਰਤਨਾਕਰ ਮਹਾਨ ਕੋਸ਼) dictionary access.

Reads data/mahankosh.sqlite as built by 00_fetch_mahankosh.py: one row per
sense, each with a part of speech, its definition text and the Gurbani
citations Bhai Kahn Singh gave for it. Homograph entries (the Kosh prints
ਸਾਰੰਗ twice, with different etymologies) are merged under one headword, senses
numbered on.

Attribution: Bhai Kahn Singh Nabha, Gur Shabad Ratnakar Mahan Kosh (1930);
digitised by redroyals/mahan-kosh-multilingual, CC BY 4.0.
"""
from __future__ import annotations
import json, os, re, sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Trailing vowel signs / nasalisation that Gurbani grammar attaches to a stem
# (ਨਾਮੁ, ਨਾਮਿ, ਨਾਮੈ ...) plus the independent vowels that close verb forms
# (ਹਰਿਓ, ਕੀਆ). Stripping one of them is a first-pass stemmer, nothing more.
INFLECTIONAL_SUFFIXES = re.compile(
    r"[ੁੂਿੀੇੈੋੌਂੰੱਓਆਏਈਉ]$"
)

# What a stripped matra suggests about the word's role (Prof. Sahib Singh's
# Gurbani Viyakaran, in the broadest strokes): aunkar on a masculine singular
# noun, sihari for locative/instrumental. Used only as a tie-break signal.
MATRA_GRAMMAR = {
    "ੁ": {"pos": "ਸੰਗ੍ਯਾ", "case": "ਕਰਤਾ (ਇਕ-ਵਚਨ)"},          # ੁ
    "ਿ": {"pos": "ਸੰਗ੍ਯਾ", "case": "ਅਧਿਕਰਣ/ਕਰਣ"},             # ਿ
    "ੇ": {"pos": "ਵਿਸ਼ੇਸ਼ਣ/ਸੰਗ੍ਯਾ", "case": "ਬਹੁ-ਵਚਨ/ਸੰਬੋਧਨ"},   # ੇ
    "ਓ": {"pos": "ਕ੍ਰਿਆ/ਵਿਸ਼ੇਸ਼ਣ", "case": "ਭੂਤਕਾਲ/ਅਵਸਥਾ"},     # ਓ
}

# The Kosh's abbreviated tags -> one canonical form each.
POS_CANON = {
    "ਸੰਗ੍ਯਾ": "ਸੰਗ੍ਯਾ", "ਸੰਗਯਾ": "ਸੰਗ੍ਯਾ",
    "ਵਿ": "ਵਿਸ਼ੇਸ਼ਣ", "ਵਿਸ਼ੇਸ਼ਣ": "ਵਿਸ਼ੇਸ਼ਣ",
    "ਕ੍ਰਿ": "ਕ੍ਰਿਆ", "ਕ੍ਰਿਆ": "ਕ੍ਰਿਆ",
    "ਕ੍ਰਿ. ਵਿ": "ਕ੍ਰਿਆ-ਵਿਸ਼ੇਸ਼ਣ", "ਕ੍ਰਿ.ਵਿ": "ਕ੍ਰਿਆ-ਵਿਸ਼ੇਸ਼ਣ", "ਕ੍ਰਿ ਵਿ": "ਕ੍ਰਿਆ-ਵਿਸ਼ੇਸ਼ਣ",
    "ਕਿਰ. ਵਿ": "ਕ੍ਰਿਆ-ਵਿਸ਼ੇਸ਼ਣ", "ਕਿਰ.ਵਿ": "ਕ੍ਰਿਆ-ਵਿਸ਼ੇਸ਼ਣ", "ਕਿਰ ਵਿ": "ਕ੍ਰਿਆ-ਵਿਸ਼ੇਸ਼ਣ",
    "ਸਰਵ": "ਸਰਵਨਾਮ", "ਵ੍ਯ": "ਅਵ੍ਯਯ", "ਅਵ੍ਯਯ": "ਅਵ੍ਯਯ",
}


@dataclass
class MahanKoshSense:
    sense_id: int
    pos: str
    definition: str
    citations: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    etymology: str = ""


@dataclass
class MahanKoshEntry:
    word: str
    senses: List[MahanKoshSense] = field(default_factory=list)

    @property
    def is_polysemous(self) -> bool:
        return len(self.senses) > 1

    def primary_definition(self) -> str:
        return self.senses[0].definition if self.senses else ""


class MahanKosh:
    """Headword -> entry, with a first-pass inflection stripper for Gurbani forms."""

    def __init__(self, db_path: Optional[str] = None):
        self.entries: Dict[str, MahanKoshEntry] = {}
        self.source: str = ""
        if db_path:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"{db_path} not found; run 00_fetch_mahankosh.py")
            self._load_sqlite(db_path)

    @classmethod
    def from_dict(cls, data: Dict[str, dict]) -> "MahanKosh":
        """Build from {word: {"senses": [{sense_id, pos, definition, citations, ...}]}} -- tests."""
        k = cls()
        for word, d in data.items():
            k.entries[word] = MahanKoshEntry(word=word, senses=[MahanKoshSense(**s) for s in d["senses"]])
        return k

    def _load_sqlite(self, path: str):
        con = sqlite3.connect(path)
        try:
            rows = con.execute("""
                SELECT e.hw, s.pos, s.text, s.citations
                FROM senses s JOIN entries e ON e.id = s.entry_id
                ORDER BY e.vol, e.page, e.id, s.sense_no""").fetchall()
            self.source = (con.execute("SELECT value FROM meta WHERE key='source'").fetchone() or [""])[0]
        finally:
            con.close()
        for hw, pos, text, cites in rows:
            entry = self.entries.setdefault(hw, MahanKoshEntry(word=hw))
            parsed = json.loads(cites) if cites else []
            entry.senses.append(MahanKoshSense(
                sense_id=len(entry.senses) + 1,
                pos=POS_CANON.get(pos, pos or ""),
                definition=text,
                citations=[c["text"] for c in parsed],
                sources=[c.get("src", "") for c in parsed]))

    def get(self, word: str) -> Optional[MahanKoshEntry]:
        return self.entries.get(word)

    def lookup_word(self, token: str):
        """(entry or None, stripped matra or '') -- exact headword first, then one suffix stripped."""
        if token in self.entries:
            return self.entries[token], ""
        m = INFLECTIONAL_SUFFIXES.search(token)
        if m:
            base = token[:m.start()]
            if base in self.entries:
                return self.entries[base], m.group(0)
        return None, ""

    def count(self) -> int:
        return len(self.entries)

    def sense_count(self) -> int:
        return sum(len(e.senses) for e in self.entries.values())
