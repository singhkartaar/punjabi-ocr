"""
Build artifacts/<key>.sqlite: the prose the server serves, and its citations.

  15_build_writings_db.py                                   # data/writings/units.jsonl -> artifacts/writings.sqlite
  15_build_writings_db.py --units data/writings/units-pa.jsonl   # -> artifacts/writings-pa.sqlite
  15_build_writings_db.py --units data/treatises/units.jsonl --out artifacts/treatises.sqlite

Reads a units file 14_embed_writings.py wrote and writes the database the web
app opens, named for the corpus the units carry: `writings-en` is served as
`writings`, so its database is writings.sqlite; a corpus in another language
keeps its suffix. The vectors sit under artifacts/corpora/<corpus>/ and the
server's CORPORA table pairs the two. It is its own file rather than a table
inside translations.sqlite for the same reason that file is separate from
gurbani.sqlite: it is a different kind of thing, read one passage at a time,
and a build without it still searches -- only the Writings tab goes quiet.

`unit_row` is the primary key AND the row index into units.i8, so retrieval
returns integers that index straight into this table with no lookup table in
between. Nothing else may renumber it.

The author's prose never enters the repository (data/writings and this file
are both gitignored: provenance and size, not secrecy). It reaches a reader
as whole passages -- through search, at most a handful per query and only the
nearest in meaning, and through Ask, as the passages an answer used -- and is
published as a data pack of the public search API. The owner's position is
that these writings are offered as a service; NOTICE.md in the public
repository records the basis per author.
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paths import ARTIFACTS, ROOT
from lib.writings_works import db_name

UNITS = os.path.join(ROOT, "data", "writings", "units.jsonl")

SCHEMA = """
CREATE TABLE works (
  work_id      TEXT PRIMARY KEY,
  title        TEXT NOT NULL,
  title_en     TEXT,               -- set where the title is Gurmukhi; NULL for an English work
  author       TEXT NOT NULL,
  folder       TEXT NOT NULL,      -- root | English | Punjabi
  original     INTEGER NOT NULL,   -- 1 where the author wrote it in English
  quote_policy TEXT NOT NULL,      -- verbatim | summarise
  parts        TEXT NOT NULL,      -- JSON array of part numbers
  files        INTEGER NOT NULL,   -- how many PDFs this work was assembled from
  units        INTEGER NOT NULL,
  language     TEXT NOT NULL DEFAULT 'en',   -- pa | en | hi
  licence      TEXT                          -- public-domain | copyright | NULL (unknown)
);
CREATE TABLE units (
  unit_row  INTEGER PRIMARY KEY,   -- also the row in units.i8; never renumber
  unit_id   TEXT NOT NULL,
  work_id   TEXT NOT NULL,
  part      INTEGER,
  page      INTEGER NOT NULL,
  para_no   INTEGER NOT NULL,
  marker    TEXT,                  -- the essay's own page label, "L127.3"
  text      TEXT NOT NULL,
  text_src  TEXT                   -- the original where `text` is a machine translation of it
);
CREATE TABLE citations (
  unit_row  INTEGER NOT NULL,
  shabad_id INTEGER NOT NULL,
  line_id   INTEGER NOT NULL,
  ang       INTEGER NOT NULL,
  score     REAL NOT NULL,
  method    TEXT NOT NULL,
  span      TEXT NOT NULL          -- the English as the essay quoted it
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_units_work ON units(work_id, part, page, para_no);
CREATE INDEX idx_citations_unit ON citations(unit_row);
CREATE INDEX idx_citations_shabad ON citations(shabad_id);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--out", help="default: artifacts/<key>.sqlite, the corpus's name without its -en")
    args = ap.parse_args()

    if not os.path.exists(args.units):
        sys.exit("no %s; run 14_embed_writings.py first" % args.units)
    with open(args.units, encoding="utf-8") as fh:
        head = json.loads(fh.readline())["_meta"]
        units = [json.loads(line) for line in fh]
    if not args.out:
        args.out = os.path.join(ARTIFACTS, db_name(head.get("corpus", "writings-en")))

    # Rebuilt whole, never patched -- but by dropping the tables inside the
    # file rather than deleting the file. On Windows a preview server holding
    # the database open makes os.remove raise, and the failure lands exactly
    # where it does most harm: after 14_embed_writings.py has already written
    # new vectors, leaving unit_row addressing rows that are no longer there.
    # A reader does not block a write, so this rebuild always completes.
    con = sqlite3.connect(args.out)
    for table in ("works", "units", "citations", "meta"):
        con.execute("DROP TABLE IF EXISTS %s" % table)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(w["work"], w["title"], w.get("title_en"), w["author"], w.get("folder", "root"),
          int(bool(w.get("original"))), w.get("quote_policy", "summarise"),
          json.dumps(w.get("parts", [])), len(w.get("files") or []), w.get("units", 0),
          w.get("language", "en"), w.get("licence"))
         for w in head["works"]])
    con.executemany(
        "INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?)",
        [(u["unit_row"], u["unit_id"], u["work"], u["part"], u["page"],
          u["para_no"], u.get("marker"), u["text"], u.get("text_src")) for u in units])
    con.executemany(
        "INSERT INTO citations VALUES (?,?,?,?,?,?,?)",
        [(u["unit_row"], c["shabad_id"], c["line_id"], c["ang"], c["score"], c["method"], c["span"])
         for u in units for c in u["cites"]])
    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("corpus", head.get("corpus", "writings-en")),
        ("author", units[0]["author"] if units else ""),
        ("authors", json.dumps(sorted({u["author"] for u in units}), ensure_ascii=False)),
        ("units", str(len(units))),
        ("works", str(len(head["works"]))),
        ("citations", str(sum(len(u["cites"]) for u in units))),
        ("built", __import__("datetime").date.today().isoformat()),
    ])
    con.commit()
    con.execute("VACUUM")
    con.close()

    size = os.path.getsize(args.out) / 1e6
    print("%d works, %d units, %d citations -> %s (%.1f MB)"
          % (len(head["works"]), len(units),
             sum(len(u["cites"]) for u in units), args.out, size))


if __name__ == "__main__":
    main()
