"""
Every parsed notation book -> artifacts/notations.sqlite, the database the web app serves.

  32_build_notations_db.py                                   # every book under data/notations/
  32_build_notations_db.py --book gurmat-sangeet-sagar-1 --allow-unmeasured
  32_build_notations_db.py --gurbani artifacts/gurbani.sqlite --urls data/notations/*/images.urls.json --require-urls

Reads data/notations/<book>/notations.jsonl and images.json (29_notation_parse.py),
applies the book's gold (data/ocr/<book>/gt/notation-gold.jsonl) so verified
records say so, and writes the tables of lib/notation_columns.json:
raags and taals from the vocabulary, authors, books, notations (one row per
record, the parsed grid as JSON), images (full crop and thumbnail, with the
release URL once tools/publish-notation-images.mjs has stamped it), and
shabad_counts for the search rows' "Notations (n)" chip.

A book goes in only when its review passed (data/raw/notation-eval-<book>.json
from 31_notation_eval.py says passed) or --allow-unmeasured is given; with
--gurbani every shabad_id is checked against the corpus; with --require-urls
an image without a URL refuses the build (the app then never shows a broken
picture). artifacts/notations-images.json lists every image with its sha256
for the upload tool.
"""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import notation
from lib.notation import PARSER_VERSION, SCHEMA_VERSION, apply_gold, read_jsonl, validate
from lib.notation_review import read_ledger
from lib.notation_render import text_english
from lib.notation_vocab import RAAGS, TAALS, VERSION as VOCAB_VERSION
from lib.paths import ARTIFACTS, NOTATIONS_DIR, OCR_DIR, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS_PATH = os.path.join(HERE, "lib", "notation_columns.json")

DDL = """
CREATE TABLE meta    (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE raags   (key TEXT PRIMARY KEY, gurmukhi TEXT, english TEXT NOT NULL,
                      in_ggs INTEGER NOT NULL, parent_key TEXT, ggs_order INTEGER, aliases TEXT);
CREATE TABLE taals   (key TEXT PRIMARY KEY, gurmukhi TEXT, english TEXT NOT NULL, matras INTEGER NOT NULL,
                      vibhag TEXT NOT NULL, sam INTEGER NOT NULL, tali TEXT NOT NULL, khali TEXT NOT NULL);
CREATE TABLE authors (author_key TEXT PRIMARY KEY, name TEXT NOT NULL, name_gurmukhi TEXT);
CREATE TABLE books   (book_key TEXT PRIMARY KEY, title TEXT NOT NULL, title_en TEXT, author_key TEXT NOT NULL,
                      part INTEGER, publisher TEXT, year INTEGER, source_url TEXT, style TEXT NOT NULL,
                      pages INTEGER NOT NULL, notations INTEGER NOT NULL, resolved INTEGER NOT NULL,
                      verified INTEGER NOT NULL, images_bytes INTEGER NOT NULL, passed_bar INTEGER NOT NULL, built TEXT NOT NULL,
                      accepted INTEGER NOT NULL DEFAULT 0, backlog INTEGER NOT NULL DEFAULT 0);
CREATE TABLE notations (
  notation_id TEXT PRIMARY KEY, book_key TEXT NOT NULL, author_key TEXT NOT NULL, ordinal INTEGER NOT NULL,
  page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, pages TEXT NOT NULL,
  kind TEXT NOT NULL,
  shabad_id INTEGER, shabad_source TEXT, ang INTEGER, first_line TEXT, writer TEXT, line_ids TEXT,
  raag_shabad TEXT, raag_shabad_key TEXT,
  raag_used TEXT, raag_used_key TEXT, raag_used_parent_key TEXT, raag_differs INTEGER NOT NULL DEFAULT 0,
  taal TEXT, taal_key TEXT, matras INTEGER, laya TEXT, partaal INTEGER NOT NULL DEFAULT 0,
  heading TEXT, sections INTEGER NOT NULL, beats INTEGER NOT NULL,
  resolve_method TEXT NOT NULL, resolve_score REAL, confidence REAL NOT NULL, confidences TEXT NOT NULL,
  verified INTEGER NOT NULL DEFAULT 0, flags TEXT NOT NULL, sargam_en TEXT,
  image_count INTEGER NOT NULL, grid TEXT,
  review_status TEXT, review_comment TEXT, review_key TEXT);  -- the ledger's verdict as built: accepted | backlog | NULL (never looked at)
CREATE TABLE images  (notation_id TEXT NOT NULL, n INTEGER NOT NULL, kind TEXT NOT NULL,
                      role TEXT NOT NULL, page INTEGER NOT NULL, path TEXT NOT NULL, url TEXT,
                      bbox TEXT, w INTEGER, h INTEGER, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
                      PRIMARY KEY (notation_id, n, kind));
CREATE TABLE shabad_counts (shabad_id INTEGER PRIMARY KEY, n INTEGER NOT NULL);
CREATE TABLE raag_notes (book_key TEXT NOT NULL, n INTEGER NOT NULL, raag_key TEXT, raag_printed TEXT,
                         page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, pages TEXT NOT NULL, heading TEXT,
                         text TEXT, image_path TEXT, image_url TEXT, sha256 TEXT, bytes INTEGER,
                         PRIMARY KEY (book_key, n));
CREATE INDEX idx_raag_notes_raag ON raag_notes(raag_key);
CREATE INDEX idx_not_shabad ON notations(shabad_id);
CREATE INDEX idx_not_raag_used ON notations(raag_used_key, ang);
CREATE INDEX idx_not_raag_shabad ON notations(raag_shabad_key, ang);
CREATE INDEX idx_not_taal ON notations(taal_key, ang);
CREATE INDEX idx_not_book ON notations(book_key, ordinal);
CREATE INDEX idx_not_author ON notations(author_key, book_key, ordinal);
"""
TABLES = ("meta", "raags", "taals", "authors", "books", "notations", "images", "shabad_counts", "raag_notes")


def j(v) -> str | None:
    return None if v is None else json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def load_gold(book: str) -> dict[str, dict]:
    p = os.path.join(OCR_DIR, book, "gt", "notation-gold.jsonl")
    out: dict[str, dict] = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    g = json.loads(line)
                    out[g["notation_id"]] = g
    return out


def eval_verdict(book: str) -> dict | None:
    p = os.path.join(ROOT, "data", "raw", "notation-eval-%s.json" % book)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_urls(paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            out.update(data.get("urls", data) if isinstance(data, dict) else {})
            # {"<notation_id>|<n>|<kind>": url}: an image whose hash this machine cannot know (a thumbnail cut on
            # another machine, recorded before thumbnails carried their hash) still finds its published URL
            if isinstance(data, dict):
                for k, v in (data.get("keys") or {}).items():
                    out["key:" + k] = v
    return out


def sha256_of(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_beats(rec: dict) -> int:
    return sum(len(line.get("beats") or []) for sec in rec.get("sections") or [] for line in sec.get("lines") or [])


def raag_row(key: str | None) -> dict:
    return (RAAGS.get(key) if isinstance(RAAGS, dict) else next((r for r in RAAGS if r["key"] == key), None)) or {}


def taal_row(key: str | None) -> dict:
    return (TAALS.get(key) if isinstance(TAALS, dict) else next((t for t in TAALS if t["key"] == key), None)) or {}


def notation_row(rec: dict, ordinal: int) -> dict:
    h = rec.get("heading") or {}
    review = rec.get("review") or {}
    sh = rec.get("shabad") or {}
    raag = h.get("raag") or {}
    taal = h.get("taal") or {}
    sections = rec.get("sections") or []
    partaal = any(s.get("taal") for s in sections) or any(s.get("taal") for s in (rec.get("layout") or {}).get("sections") or [])
    conf = {"shabad": sh.get("confidence"), "raag": raag.get("confidence"), "taal": taal.get("confidence")}
    return {
        "notation_id": rec["notation_id"], "book_key": rec["book_key"], "author_key": rec["source"]["author_key"],
        "ordinal": ordinal, "page_start": min(rec["pages"]), "page_end": max(rec["pages"]), "pages": j(rec["pages"]),
        "kind": rec["kind"],
        # an ang only for a shabad the corpus named: a number read off an unidentified page is no ang
        "shabad_id": sh.get("shabad_id"), "shabad_source": sh.get("source"), "ang": sh.get("ang") if sh.get("shabad_id") is not None else None,
        "first_line": sh.get("first_line"), "writer": sh.get("writer"), "line_ids": j(sh.get("line_ids") or []),
        "raag_shabad": sh.get("raag") or raag_row(rec.get("raag_shabad")).get("en"), "raag_shabad_key": rec.get("raag_shabad"),
        "raag_used": raag.get("printed"), "raag_used_key": raag.get("key"), "raag_used_parent_key": raag.get("parent_key"),
        "raag_differs": 1 if rec.get("raag_differs") else 0,
        "taal": taal.get("printed"), "taal_key": taal.get("key"), "matras": taal.get("matras") or taal_row(taal.get("key")).get("matras"),
        "laya": taal.get("laya"), "partaal": 1 if partaal else 0,
        "heading": h.get("raw"), "sections": len(sections) or len((rec.get("layout") or {}).get("sections") or []),
        "beats": count_beats(rec),
        "resolve_method": sh.get("method") or "none", "resolve_score": sh.get("confidence"),
        "confidence": float(sh.get("confidence") or 0.0), "confidences": j(conf) or "{}",
        "verified": 1 if rec.get("verified") else 0, "flags": j(rec.get("flags") or []) or "[]",
        "sargam_en": text_english(rec) if sections else None,
        "image_count": len(rec.get("images") or []), "grid": j(sections) if sections else None,
        # the review as it stood when this was built, so a build that ships everything (auto) still says
        # what was accepted, what was commented and what nobody has looked at; the ledger is the record
        "review_status": review.get("status"), "review_comment": review.get("comment") or None,
        "review_key": rec.get("review_key") or review.get("key"),
    }


def image_rows(rec: dict, book_dir: str, urls: dict[str, str], by_file: dict[str, dict]) -> list[dict]:
    rows = []
    key = lambda n, kind: urls.get("key:%s|%s|%s" % (rec["notation_id"], n, kind))
    for im in rec.get("images") or []:
        rows.append({"notation_id": rec["notation_id"], "n": im["n"], "kind": "full", "role": im["role"], "page": im["page"],
                     "path": rec["book_key"] + "/" + im["file"], "url": urls.get(im["sha256"]) or key(im["n"], "full"),
                     "bbox": j(im.get("bbox")), "w": im.get("w"), "h": im.get("h"), "bytes": im["bytes"], "sha256": im["sha256"]})
        if im.get("thumb"):
            tpath = os.path.join(book_dir, im["thumb"])
            known = by_file.get(im["thumb"]) or {}
            sha = im.get("thumb_sha256") or known.get("sha256") or (sha256_of(tpath) if os.path.exists(tpath) else "")
            size = im.get("thumb_bytes") or known.get("bytes") or (os.path.getsize(tpath) if os.path.exists(tpath) else 0)
            rows.append({"notation_id": rec["notation_id"], "n": im["n"], "kind": "thumb", "role": im["role"], "page": im["page"],
                         "path": rec["book_key"] + "/" + im["thumb"], "url": (urls.get(sha) if sha else None) or key(im["n"], "thumb"),
                         "bbox": j(im.get("bbox")), "w": known.get("w"), "h": known.get("h"), "bytes": size, "sha256": sha})
    return rows


def build(src: str, out: str, books: list[str] | None, gurbani: str | None, urls: dict[str, str],
          require_urls: bool, allow_unmeasured: bool, verify_images: bool, release_base: str | None,
          accepted_only: bool = False) -> dict:
    with open(COLUMNS_PATH, encoding="utf-8") as fh:
        columns = json.load(fh)["tables"]
    dirs = sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.exists(os.path.join(d, "notations.jsonl")))
    if books:
        dirs = [d for d in dirs if os.path.basename(d) in books]
        missing = set(books) - {os.path.basename(d) for d in dirs}
        if missing:
            sys.exit("no notations.jsonl for %s under %s" % (", ".join(sorted(missing)), src))
    if not dirs:
        sys.exit("nothing to build: no <book>/notations.jsonl under %s" % src)

    con_g = sqlite3.connect(gurbani) if gurbani else None
    known_shabads: set[int] | None = None
    if con_g is not None:
        known_shabads = {r[0] for r in con_g.execute("SELECT shabad_id FROM shabads")}

    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(DDL)
    for t in TABLES:
        have = [r[1] for r in con.execute("PRAGMA table_info(%s)" % t)]
        if have != columns[t]:
            sys.exit("%s: columns %s differ from lib/notation_columns.json %s" % (t, have, columns[t]))

    def insert(table: str, row: dict) -> None:
        cols = columns[table]
        con.execute("INSERT OR REPLACE INTO %s (%s) VALUES (%s)" % (table, ",".join(cols), ",".join("?" * len(cols))),
                    [row.get(c) for c in cols])

    raag_iter = RAAGS.values() if isinstance(RAAGS, dict) else RAAGS
    ggs = [r["key"] for r in raag_iter if r.get("in_ggs") and r.get("kind", "raag") == "raag" and not r.get("parent")]
    for r in (RAAGS.values() if isinstance(RAAGS, dict) else RAAGS):
        insert("raags", {"key": r["key"], "gurmukhi": (r.get("pa") or [None])[0] if isinstance(r.get("pa"), list) else r.get("pa"),
                         "english": (r.get("en") or [r["key"]])[0] if isinstance(r.get("en"), list) else (r.get("en") or r["key"]),
                         "in_ggs": 1 if r.get("in_ggs") else 0, "parent_key": r.get("parent"),
                         "ggs_order": (ggs.index(r["key"]) + 1) if r["key"] in ggs else None,
                         "aliases": j({"pa": r.get("pa") or [], "en": r.get("en") or []})})
    for t in (TAALS.values() if isinstance(TAALS, dict) else TAALS):
        insert("taals", {"key": t["key"], "gurmukhi": (t.get("pa") or [None])[0] if isinstance(t.get("pa"), list) else t.get("pa"),
                         "english": (t.get("en") or [t["key"]])[0] if isinstance(t.get("en"), list) else (t.get("en") or t["key"]),
                         "matras": int(t.get("matras") or 0), "vibhag": j(t.get("vibhag") or []) or "[]",
                         "sam": int(t.get("sam") or 1), "tali": j(t.get("tali") or []) or "[]", "khali": j(t.get("khali") or []) or "[]"})

    counts: dict[int, int] = {}
    images_manifest: list[dict] = []
    totals = {"books": 0, "notations": 0, "images": 0, "published": 0, "verified": 0, "resolved": 0}
    problems: list[str] = []
    skipped: list[str] = []
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parser_majors: set[str] = set()
    for d in dirs:
        book = os.path.basename(d)
        meta, records = read_jsonl(os.path.join(d, "notations.jsonl"))
        parser_majors.add(str(meta.get("parser_version", PARSER_VERSION)).split(".")[0])
        verdict = eval_verdict(book)
        ledger = read_ledger(book)
        n_acc = sum(1 for e in ledger.values() if e["status"] == "accepted")
        n_bl = sum(1 for e in ledger.values() if e["status"] == "backlog")
        passed = bool(verdict and verdict.get("passed")) or (accepted_only and n_acc > 0)
        if accepted_only:
            records = [r for r in records if r.get("verified")]           # the reviewed notations only
            if not records:
                skipped.append(book)                                      # nothing accepted yet: not a problem, just not shipped
                continue
            passed = True                                                 # the acceptance is the review
        if not passed and not allow_unmeasured:
            problems.append("%s: %s; review it (30/31) or pass --allow-unmeasured"
                            % (book, "review under the bar: " + "; ".join(verdict.get("fails") or []) if verdict else "no review yet"))
            continue
        gold = load_gold(book)
        by_file: dict[str, dict] = {}
        ij = os.path.join(d, "images.json")
        if os.path.exists(ij):
            with open(ij, encoding="utf-8") as fh:
                by_file = {f["file"]: f for f in json.load(fh).get("files", [])}
        book_urls = dict(urls)
        bu = os.path.join(d, "images.urls.json")
        if os.path.exists(bu):
            book_urls.update(load_urls([bu]))
        author = meta.get("author") or "Unknown"
        author_key = records[0]["source"]["author_key"] if records else notation.notation_slug(author)
        insert("authors", {"author_key": author_key, "name": author, "name_gurmukhi": meta.get("author_gurmukhi")})
        n_res = n_ver = n_img = 0
        img_bytes = 0
        for k, rec in enumerate(records):
            apply_gold(rec, gold.get(rec["notation_id"]))
            errs = validate(rec)
            if errs:
                problems.append("%s: invalid after gold: %s" % (rec["notation_id"], errs[:3]))
                continue
            sid = (rec.get("shabad") or {}).get("shabad_id")
            if sid is not None and known_shabads is not None and sid not in known_shabads:
                problems.append("%s: shabad_id %s is not in %s" % (rec["notation_id"], sid, gurbani))
                continue
            row = notation_row(rec, k + 1)
            if row["raag_used_key"] and not raag_row(row["raag_used_key"]):
                problems.append("%s: raag key %r unknown to the vocabulary" % (rec["notation_id"], row["raag_used_key"]))
                continue
            if row["taal_key"] and not taal_row(row["taal_key"]):
                problems.append("%s: taal key %r unknown to the vocabulary" % (rec["notation_id"], row["taal_key"]))
                continue
            insert("notations", row)
            ims = list(image_rows(rec, d, book_urls, by_file))
            # the app shows a notation's block cuts (and their thumbs), or every image of one without blocks;
            # those are what tools/publish-notation-images.mjs publishes, and all --require-urls asks for
            has_block = any(im["role"] == "block" for im in ims)
            for im in ims:
                full = os.path.join(src, im["path"])
                if verify_images and os.path.exists(full) and im["sha256"] and sha256_of(full) != im["sha256"]:
                    problems.append("%s: %s does not match its sha256" % (rec["notation_id"], im["path"]))
                if require_urls and not im["url"] and (im["role"] == "block" or not has_block):
                    problems.append("%s: image %s has no URL" % (rec["notation_id"], im["path"]))
                insert("images", im)
                images_manifest.append({"notation_id": im["notation_id"], "n": im["n"], "kind": im["kind"],
                                        "path": im["path"], "sha256": im["sha256"], "bytes": im["bytes"], "url": im["url"]})
                n_img += 1
                img_bytes += im["bytes"] or 0
                if im["url"]:
                    totals["published"] += 1
            if sid is not None:
                counts[sid] = counts.get(sid, 0) + 1
                n_res += 1
            if rec.get("verified"):
                n_ver += 1
        # what the book says about its raags, beside the notations
        rp = os.path.join(d, "raags.jsonl")
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    note = json.loads(line)
                    if "_meta" in note:
                        continue
                    im = note.get("image") or {}
                    insert("raag_notes", {"book_key": book, "n": note["n"], "raag_key": (note.get("raag") or {}).get("key"),
                                          "raag_printed": (note.get("raag") or {}).get("printed"),
                                          "page_start": min(note["pages"]), "page_end": max(note["pages"]), "pages": j(note["pages"]),
                                          "heading": note.get("heading"), "text": note.get("text"),
                                          "image_path": (book + "/" + im["file"]) if im.get("file") else None,
                                          "image_url": book_urls.get(im.get("sha256")) if im.get("sha256") else None,
                                          "sha256": im.get("sha256"), "bytes": im.get("bytes")})
                    if im.get("file"):
                        images_manifest.append({"notation_id": None, "raag_note": [book, note["n"]], "n": 1, "kind": "full",
                                                "path": book + "/" + im["file"], "sha256": im.get("sha256"), "bytes": im.get("bytes"),
                                                "url": book_urls.get(im.get("sha256"))})
        insert("books", {"book_key": book, "title": meta.get("title") or book, "title_en": meta.get("title_en"),
                         "author_key": author_key, "part": meta.get("part"), "publisher": meta.get("publisher"),
                         "year": meta.get("year"), "source_url": meta.get("source_url"),
                         "style": j(meta.get("style") or {}) or "{}",
                         "pages": int(meta.get("pages") or 0), "notations": len(records), "resolved": n_res,
                         "verified": n_ver, "images_bytes": img_bytes, "passed_bar": 1 if passed else 0, "built": built,
                         "accepted": n_acc, "backlog": n_bl})
        totals["books"] += 1
        totals["notations"] += len(records)
        totals["images"] += n_img
        totals["verified"] += n_ver
        totals["resolved"] += n_res
    if skipped:
        print("%d book(s) with nothing accepted yet, not shipped: %s" % (len(skipped), ", ".join(skipped[:6]) + (" ..." if len(skipped) > 6 else "")))
    if len(parser_majors) > 1:
        problems.append("parser major versions mixed in one build: %s" % sorted(parser_majors))
    if problems:
        con.close()
        os.remove(tmp)
        for p in problems[:40]:
            print("  " + p)
        sys.exit("refusing to build: %d problem(s)" % len(problems))
    for sid, n in counts.items():
        insert("shabad_counts", {"shabad_id": sid, "n": n})
    meta_rows = {"schema": str(SCHEMA_VERSION), "parser_version": PARSER_VERSION, "vocab_version": VOCAB_VERSION,
                 "columns_version": "1", "built": built, "books": str(totals["books"]), "notations": str(totals["notations"]),
                 "shabads": str(len(counts)), "images": str(totals["images"]), "images_published": str(totals["published"]),
                 "images_release_base": release_base or "", "bars": j(_bars()),
                 # accepted: only reviewed notations (manual); all: every record, with its review state per row (auto)
                 "review_mode": "accepted" if accepted_only else "all"}
    for k, v in meta_rows.items():
        insert("meta", {"key": k, "value": v})
    con.commit()
    con.execute("VACUUM")
    con.close()
    if os.path.exists(out):
        os.remove(out)
    os.replace(tmp, out)
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(out)), "notations-images.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"built": built, "src": os.path.relpath(src, ROOT).replace("\\", "/"), "images": images_manifest},
                  fh, ensure_ascii=False, indent=0)
    return {**totals, "shabads": len(counts), "out": out, "images_manifest": manifest_path}


def _bars() -> dict:
    """The acceptance bars, read from 31_notation_eval.py without importing a module whose name starts with a digit."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("notation_eval", os.path.join(HERE, "31_notation_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BARS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=NOTATIONS_DIR, help="data/notations (NOTATIONS_DIR)")
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "notations.sqlite"))
    ap.add_argument("--book", action="append", help="only this book (repeatable)")
    ap.add_argument("--gurbani", help="artifacts/gurbani.sqlite: every shabad_id must exist in it")
    ap.add_argument("--urls", action="append", default=[], help="JSON {sha256: url} file(s) or globs from the upload tool")
    ap.add_argument("--require-urls", action="store_true", help="refuse an image without a URL")
    ap.add_argument("--allow-unmeasured", action="store_true", help="build a book whose review has not passed")
    ap.add_argument("--accepted-only", action="store_true", help="only the notations the review ledger accepted (the public pack's default once a book is clear)")
    ap.add_argument("--verify-images", action="store_true", help="re-hash every crop against its record")
    ap.add_argument("--release-base", help="the URL prefix the images are published under (recorded in meta)")
    args = ap.parse_args()
    got = build(args.src, args.out, args.book, args.gurbani, load_urls(args.urls), args.require_urls,
                args.allow_unmeasured, args.verify_images, args.release_base, args.accepted_only)
    print("%(books)d book(s), %(notations)d notation(s), %(resolved)d with a shabad (%(shabads)d shabads), "
          "%(verified)d verified, %(images)d image(s) (%(published)d with a URL) -> %(out)s; %(images_manifest)s" % got)


if __name__ == "__main__":
    main()
