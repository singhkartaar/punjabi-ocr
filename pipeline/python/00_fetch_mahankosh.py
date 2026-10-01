"""
Fetch Bhai Kahn Singh Nabha's Gur Shabad Ratnakar Mahan Kosh and compile it
into data/mahankosh.sqlite, one row per SENSE.

Source: redroyals/mahan-kosh-multilingual (CC BY 4.0) -- clean Unicode
Gurmukhi of the Punjabi University Patiala edition:
  core.json      {"entries": [{"id": "1-37-0", "hw": "ਉ", "vol": 1, "page": 37, ...}]}
  gurmukhi.json  {"1-37-0": {"text": "ਸੰ. उ. ਸੰਗ੍ਯਾ- ਬ੍ਰਹਮਾ। ੨. ਵਿਸਨੁ। ੩. ..."}}

Entry text format, as printed: the first sense is unnumbered; each further
sense starts with a Gurmukhi numeral and a full stop ("੨. ", "੩. "). A part of
speech tag ("ਸੰਗ੍ਯਾ-", "ਵਿ-", "ਕ੍ਰਿ-", "ਵ੍ਯ-" ...) applies until the next one.
Scriptural citations are quoted, followed by a source in parentheses:
  "ਸਹਜੇ ਗਾਵਿਆ ਥਾਇ ਪਵੈ." (ਸ੍ਰੀ ਅਃ ਮਃ ੩)
The source tags cover SGGS as well as Dasam Granth and other texts; matching a
citation against the corpus (05_eval.py) is what decides which it is.

Why one row per sense: the Kosh is the ground truth for polysemy. ਹਰਿ has 13
senses; treating the whole entry as one definition would inject all 13 into
every line containing ਹਰਿ, the exact "polysemy pollution" the disambiguator
exists to avoid.
"""
from __future__ import annotations
import argparse, json, os, re, sqlite3, sys, time, unicodedata, urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paths import MAHANKOSH_DB, MAHANKOSH_CACHE

BASE = "https://raw.githubusercontent.com/redroyals/mahan-kosh-multilingual/main/data/entries/"
FILES = ["core.json", "gurmukhi.json"]

GURMUKHI_DIGITS = "੦੧੨੩੪੫੬੭੮੯"
# A sense number: Gurmukhi numeral(s) + "." at the start of a segment. Numbers
# inside citations or sources (ਮਃ ੩) are preceded by a letter/space without a
# following ".", or by "(", and are not matched.
SENSE_NUM = re.compile(r"(?<![੦-੯(\w])([੧-੯][੦-੯]?)\.\s")
POS_TAG = re.compile(r"(ਸੰਗ੍ਯਾ|ਸੰਗਯਾ|ਵਿਸ਼ੇਸ਼ਣ|ਕ੍ਰਿ\.?\s?ਵਿ|ਕਿਰ\.?\s?ਵਿ|ਸਰਵ|ਵ੍ਯ|ਵਿ|ਕ੍ਰਿ|ਅਵ੍ਯਯ)-")
CITATION = re.compile(r"\"([^\"]+)\"\s*(?:\(([^)]*)\))?")
JUNK = re.compile("[\uE000-\uF8FF\u00B9\u00B2\u00B3\u2070-\u2079#]")   # font glyph leftovers (private use area), footnote marks, paragraph marks


def gurmukhi_int(s: str) -> int:
    return int("".join(str(GURMUKHI_DIGITS.index(c)) for c in s))


def download(name: str) -> str:
    os.makedirs(MAHANKOSH_CACHE, exist_ok=True)
    dst = os.path.join(MAHANKOSH_CACHE, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return dst
    print(f"downloading {name} ...", flush=True)
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Gurbani-MahanKosh-Indexer/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as o:
        o.write(r.read())
    return dst


def clean(text: str) -> str:
    return " ".join(JUNK.sub("", unicodedata.normalize("NFC", text)).split())


def split_senses(text: str) -> list[str]:
    """Split an entry into its numbered senses; numbers must run ੨, ੩, ੪ ... in order."""
    cuts, expect = [], 2
    for m in SENSE_NUM.finditer(text):
        if gurmukhi_int(m.group(1)) == expect:
            cuts.append(m.start())
            expect += 1
    parts, prev = [], 0
    for c in cuts:
        parts.append(text[prev:c])
        prev = c
    parts.append(text[prev:])
    return [p.strip(" ।.") for p in parts if p.strip(" ।.")]


def parse_sense(sense_text: str, inherited_pos: str):
    m = POS_TAG.search(sense_text)
    pos = m.group(1) if m else inherited_pos
    cites = [{"text": " ".join(t.split()).strip(" ."), "src": (s or "").strip()} for t, s in CITATION.findall(sense_text)]
    cites = [c for c in cites if len(c["text"]) > 3]
    return pos, cites


def build(db_path: str):
    core = json.load(open(download("core.json"), encoding="utf-8"))
    gur = json.load(open(download("gurmukhi.json"), encoding="utf-8"))
    entries = core["entries"]
    print(f"{len(entries)} entries in core.json, {len(gur)} with Gurmukhi text")

    con = sqlite3.connect(db_path)
    con.executescript("""
        DROP TABLE IF EXISTS senses; DROP TABLE IF EXISTS entries; DROP TABLE IF EXISTS mahankosh;
        CREATE TABLE entries (id TEXT PRIMARY KEY, hw TEXT NOT NULL, vol INTEGER, page INTEGER, text TEXT NOT NULL);
        CREATE TABLE senses (entry_id TEXT NOT NULL, sense_no INTEGER NOT NULL, pos TEXT, text TEXT NOT NULL,
                             citations TEXT NOT NULL, PRIMARY KEY (entry_id, sense_no));
        CREATE INDEX idx_entries_hw ON entries(hw);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    seen, n_entries, n_senses, n_cites = set(), 0, 0, 0
    t0 = time.time()
    for e in entries:
        if e.get("excluded"):
            continue
        hw = clean(e.get("hw") or "")
        text = clean((gur.get(e["id"]) or {}).get("text") or "")
        if not hw or not text:
            continue
        if (hw, text) in seen:          # page-bound duplicates in the source
            continue
        seen.add((hw, text))
        con.execute("INSERT INTO entries VALUES (?,?,?,?,?)", (e["id"], hw, e.get("vol"), e.get("page"), text))
        n_entries += 1
        pos = ""
        for i, s in enumerate(split_senses(text), start=1):
            pos, cites = parse_sense(s, pos)
            con.execute("INSERT INTO senses VALUES (?,?,?,?,?)",
                        (e["id"], i, pos, s, json.dumps(cites, ensure_ascii=False)))
            n_senses += 1
            n_cites += len(cites)
    con.execute("INSERT INTO meta VALUES ('source', ?)",
                ("Bhai Kahn Singh Nabha, Gur Shabad Ratnakar Mahan Kosh; digitised by "
                 "redroyals/mahan-kosh-multilingual (CC BY 4.0)",))
    con.execute("INSERT INTO meta VALUES ('built', ?)", (time.strftime("%Y-%m-%d"),))
    con.commit()
    con.close()
    print(f"{n_entries} entries, {n_senses} senses, {n_cites} citations -> {db_path} "
          f"({os.path.getsize(db_path)/1e6:.1f} MB) in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=MAHANKOSH_DB)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build(args.output)


if __name__ == "__main__":
    main()
