"""
The review ledger: the reviewer's verdict on each notation, kept under a
stable identity, so a tick is given once and an accepted notation is never
redone.

A notation's identity across re-runs is not its notation_id (book:page:seq
shifts when a layout rule moves a cut) but its review key:

    <book_key>/<shabad_id|none>/<raag_used_key|->/<taal_key|->#<nth>

the markers a reviewer would name -- the book (and with it the author), the
shabad, the raag it was set in, the taal -- and `nth`, the ordinal in page
order among the book's notations that share the first three (the same
shabad set twice in one raag and taal: a second notation, a partaal). The
same shabad in another raag or another book is another key by construction.

On a re-run a ledger entry re-attaches to a fresh record by overlap, not by
key alone: same book, same shabad (or both unknown), overlapping pages, and
extents on the first shared page overlapping by OVERLAP or more; among
several, the nearest nth. An entry that attaches nowhere is "lost" and is
reported, never dropped in silence.

The ledger is one append-only JSON-lines file a book under REVIEW_DIR; the
last line for a key wins. Its statuses:

    accepted   the cut and the shabad link are right: the record is frozen
               (its fixture is what later runs emit), level "cut"
    backlog    something is wrong; the comment says what; it comes back
               after the next algorithm change
    rejected   not a notation (a prose cut, a tabla table): dropped from the
               build, not shown again

A fixture is the accepted record as reviewed (`fixtures/<book>/<key-slug>.json`)
with its ledger line, and its images copied beside it (`fixtures/<book>/images/
<sha256>.png`), so the images can be restored on another machine without
re-cutting, and the regression check has something exact to compare with.
"""
from __future__ import annotations
import datetime as dt
import json
import os
import re
import shutil

from lib.paths import REVIEW_DIR

STATUSES = ("accepted", "backlog", "rejected")
LEVELS = ("cut", "grid")
OVERLAP = 0.6               # the extent overlap (on the first shared page) that re-attaches an entry
IOU_PASS = 0.9              # the extent overlap an accepted notation must keep for the check to pass


# ---- identity --------------------------------------------------------------

def key_parts(rec: dict) -> tuple[str, str, str, str]:
    h = rec.get("heading") or {}
    sh = rec.get("shabad") or {}
    sid = sh.get("shabad_id")
    return (rec["book_key"], str(sid) if sid is not None else "none",
            ((h.get("raag") or {}).get("key") or "-"), ((h.get("taal") or {}).get("key") or "-"))


def assign_keys(records: list[dict]) -> dict[str, dict]:
    """
    {review_key: record} over a book's records, in page order; `nth` numbers
    the records that share book, shabad, raag and taal. Each record gets
    `review_key` and `review_nth` set.
    """
    counts: dict[tuple, int] = {}
    out: dict[str, dict] = {}
    for rec in sorted(records, key=lambda r: (r["pages"][0] if r.get("pages") else r.get("page", 0), r.get("seq", 0))):
        parts = key_parts(rec)
        counts[parts] = counts.get(parts, 0) + 1
        rec["review_nth"] = counts[parts]
        rec["review_key"] = "%s/%s/%s/%s#%d" % (*parts, counts[parts])
        out[rec["review_key"]] = rec
    return out


def key_slug(key: str) -> str:
    """A review key as a file name: 'book/913/bhairavi/dadra#1' -> 'book__913__bhairavi__dadra__1'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key.replace("/", "__").replace("#", "__"))


def extent_of(rec: dict) -> dict[int, list[int]]:
    """{page: [x0, y0, x1, y1]} from the record's layout extents (or its block images)."""
    ext = {}
    for k, v in ((rec.get("layout") or {}).get("extent") or {}).items():
        ext[int(k)] = [int(x) for x in v]
    if not ext:
        for im in rec.get("images") or []:
            if im.get("role") == "block" and im.get("bbox"):
                ext[int(im["page"])] = [int(x) for x in im["bbox"]]
    return ext


def _overlap_1d(a: list[int], b: list[int]) -> float:
    """Vertical overlap of two extents as a share of the shorter one (the cut is margin to margin, height is what moves)."""
    lo, hi = max(a[1], b[1]), min(a[3], b[3])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1, min(a[3] - a[1], b[3] - b[1]))


def entry_of(rec: dict, status: str, comment: str = "", by: str = "", round_: int | None = None, level: str = "cut") -> dict:
    """A ledger line for this record and verdict."""
    h = rec.get("heading") or {}
    sh = rec.get("shabad") or {}
    return {
        "key": rec["review_key"], "level": level, "status": status,
        "notation_id": rec["notation_id"], "book_key": rec["book_key"], "author_key": (rec.get("source") or {}).get("author_key"),
        "shabad_id": sh.get("shabad_id"), "raag_used_key": (h.get("raag") or {}).get("key"),
        "raag_shabad_key": rec.get("raag_shabad"), "taal_key": (h.get("taal") or {}).get("key"), "nth": rec.get("review_nth"),
        "pages": list(rec.get("pages") or []), "extent": {str(k): v for k, v in extent_of(rec).items()},
        "images": [{"file": im["file"], "sha256": im["sha256"], "role": im["role"], "page": im["page"]} for im in rec.get("images") or []],
        "content_hash": (rec.get("source") or {}).get("content_hash"),
        "parser_version": ((rec.get("source") or {}).get("parser") or {}).get("version"),
        "commit": (rec.get("source") or {}).get("commit"),
        "comment": comment or "", "by": by or os.environ.get("USER") or os.environ.get("USERNAME") or "",
        "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "round": round_,
    }


# ---- the ledger ------------------------------------------------------------

def ledger_path(book: str, review_dir: str | None = None) -> str:
    return os.path.join(review_dir or REVIEW_DIR, book + ".jsonl")


def read_ledger(book: str, review_dir: str | None = None) -> dict[str, dict]:
    """{key: last entry} for the book; an empty dict when there is no ledger yet."""
    path = ledger_path(book, review_dir)
    out: dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("level", "cut") == "cut":
                out[e["key"]] = e
    return out


def read_ledger_lines(book: str, review_dir: str | None = None) -> list[dict]:
    path = ledger_path(book, review_dir)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def append_entry(entry: dict, review_dir: str | None = None) -> str:
    path = ledger_path(entry["book_key"], review_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


# ---- fixtures --------------------------------------------------------------

def fixture_dir(book: str, review_dir: str | None = None) -> str:
    return os.path.join(review_dir or REVIEW_DIR, "fixtures", book)


def fixture_path(entry: dict, review_dir: str | None = None) -> str:
    return os.path.join(fixture_dir(entry["book_key"], review_dir), key_slug(entry["key"]) + ".json")


def save_fixture(rec: dict, entry: dict, images_dir: str, review_dir: str | None = None) -> str:
    """The accepted record and its ledger line as a fixture; its images copied beside it by sha256."""
    path = fixture_path(entry, review_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_dir = os.path.join(fixture_dir(entry["book_key"], review_dir), "images")
    os.makedirs(img_dir, exist_ok=True)
    for im in rec.get("images") or []:
        src = os.path.join(images_dir, os.path.basename(im["file"]))
        dst = os.path.join(img_dir, im["sha256"] + ".png")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
        thumb = im.get("thumb")
        if thumb:
            tsrc = os.path.join(images_dir, os.path.basename(thumb))
            tdst = os.path.join(img_dir, im["sha256"] + ".thumb.png")
            if os.path.exists(tsrc) and not os.path.exists(tdst):
                shutil.copyfile(tsrc, tdst)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"entry": entry, "record": rec}, fh, ensure_ascii=False, indent=1)
    return path


def load_fixture(entry: dict, review_dir: str | None = None) -> dict | None:
    path = fixture_path(entry, review_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def restore_images(rec: dict, entry: dict, images_dir: str, review_dir: str | None = None) -> int:
    """The fixture's images copied back under the record's file names where they are missing. Returns how many."""
    img_dir = os.path.join(fixture_dir(entry["book_key"], review_dir), "images")
    n = 0
    os.makedirs(images_dir, exist_ok=True)
    for im in rec.get("images") or []:
        dst = os.path.join(images_dir, os.path.basename(im["file"]))
        src = os.path.join(img_dir, im["sha256"] + ".png")
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copyfile(src, dst)
            n += 1
        if im.get("thumb"):
            tdst = os.path.join(images_dir, os.path.basename(im["thumb"]))
            tsrc = os.path.join(img_dir, im["sha256"] + ".thumb.png")
            if not os.path.exists(tdst) and os.path.exists(tsrc):
                shutil.copyfile(tsrc, tdst)
    return n


# ---- matching and applying -------------------------------------------------

def match_entry(entry: dict, records: list[dict]) -> dict | None:
    """The fresh record a ledger entry re-attaches to, by shabad, pages and extent overlap; None when nothing does."""
    sid = entry.get("shabad_id")
    pages = set(entry.get("pages") or [])
    ext = {int(k): v for k, v in (entry.get("extent") or {}).items()}
    # the key itself first: two notations of one shabad on overlapping pages (a second taal, an
    # inherited verse) are told apart by their nth, and an entry must not take its neighbour
    for rec in records:
        if rec.get("review_key") == entry["key"] and (pages & set(rec.get("pages") or [])):
            return rec
    if any(rec.get("review_key") == entry["key"] for rec in records):
        return None                                        # the key exists but moved off its pages: not the same notation
    best, best_score = None, 0.0
    for rec in records:
        if rec["book_key"] != entry["book_key"]:
            continue
        if (rec.get("shabad") or {}).get("shabad_id") != sid:
            continue
        shared = sorted(pages & set(rec.get("pages") or []))
        if not shared:
            continue
        rext = extent_of(rec)
        p = shared[0]
        ov = _overlap_1d(ext[p], rext[p]) if p in ext and p in rext else 0.0
        if ov < OVERLAP:
            continue
        score = ov + (0.5 if rec.get("review_nth") == entry.get("nth") else 0.0) + 0.1 * len(shared)
        if score > best_score:
            best, best_score = rec, score
    return best


def attach(ledger: dict[str, dict], records: list[dict]) -> dict[str, dict]:
    """{notation_id: entry} -- every ledger entry attached to at most one record, keys first, then overlap, no record taken twice."""
    out: dict[str, dict] = {}
    taken: set[str] = set()
    for key, entry in sorted(ledger.items()):
        rec = next((r for r in records if r.get("review_key") == key and set(entry.get("pages") or []) & set(r.get("pages") or [])), None)
        if rec is not None and rec["notation_id"] not in taken:
            out[rec["notation_id"]] = entry
            taken.add(rec["notation_id"])
    for key, entry in sorted(ledger.items()):
        if any(e is entry for e in out.values()):
            continue
        rec = match_entry(entry, [r for r in records if r["notation_id"] not in taken])
        if rec is not None:
            out[rec["notation_id"]] = entry
            taken.add(rec["notation_id"])
    return out


def iou(a: dict[int, list[int]], b: dict[int, list[int]]) -> float:
    """The overlap of two per-page extents over their union, by height summed over pages."""
    inter = union = 0
    for p in set(a) | set(b):
        if p in a and p in b:
            lo, hi = max(a[p][1], b[p][1]), min(a[p][3], b[p][3])
            inter += max(0, hi - lo)
            union += max(a[p][3], b[p][3]) - min(a[p][1], b[p][1])
        else:
            e = a.get(p) or b.get(p)
            union += e[3] - e[1]
    return inter / union if union else 0.0


def drift_of(entry: dict, fresh: dict | None) -> dict:
    """What a re-run would change about an accepted notation: pages, shabad, extent overlap."""
    before = {"pages": entry.get("pages") or [], "shabad_id": entry.get("shabad_id")}
    if fresh is None:
        return {"key": entry["key"], "lost": True, **{"pages_before": before["pages"], "shabad_before": before["shabad_id"]}}
    ext_a = {int(k): v for k, v in (entry.get("extent") or {}).items()}
    ext_b = extent_of(fresh)
    return {"key": entry["key"], "lost": False,
            "pages_before": before["pages"], "pages_after": list(fresh.get("pages") or []),
            "shabad_before": before["shabad_id"], "shabad_after": (fresh.get("shabad") or {}).get("shabad_id"),
            "extent_iou": round(iou(ext_a, ext_b), 3),
            "same": (list(fresh.get("pages") or []) == before["pages"] and (fresh.get("shabad") or {}).get("shabad_id") == before["shabad_id"]
                     and iou(ext_a, ext_b) >= IOU_PASS)}


def apply_review(records: list[dict], ledger: dict[str, dict], images_dir: str, review_dir: str | None = None) -> dict:
    """
    The ledger applied to a run's records (29_notation_parse.py, after the
    records are built): accepted entries replace their fresh records with
    the fixture (verified, images restored), lost ones are emitted from the
    fixture anyway, backlog ones annotate the fresh record with the
    comment, rejected ones drop it. Returns {"records", "accepted",
    "reused", "lost", "backlog", "rejected", "drift"}; `records` is the
    list to write.
    """
    assign_keys(records)
    out = {"accepted": 0, "reused": 0, "lost": 0, "backlog": 0, "rejected": 0, "drift": []}
    taken: set[int] = set()
    replaced: dict[int, dict] = {}
    extra: list[dict] = []
    attached = attach(ledger, records)
    by_entry = {id(e): nid for nid, e in attached.items()}
    by_nid = {r["notation_id"]: r for r in records}
    for key, entry in sorted(ledger.items()):
        fresh = by_nid.get(by_entry.get(id(entry)))
        if entry["status"] == "accepted":
            out["accepted"] += 1
            fx = load_fixture(entry, review_dir)
            frozen = (fx or {}).get("record")
            d = drift_of(entry, fresh)
            out["drift"].append(d)
            if frozen is None:
                continue                                   # no fixture on this machine: the fresh record stands
            frozen = json.loads(json.dumps(frozen))
            frozen["verified"] = True
            frozen["review"] = {"key": key, "status": "accepted", "at": entry.get("at"), "round": entry.get("round")}
            restore_images(frozen, entry, images_dir, review_dir)
            if fresh is not None:
                idx = records.index(fresh)
                taken.add(idx)
                replaced[idx] = frozen
                out["reused"] += 1
            else:
                out["lost"] += 1
                extra.append(frozen)
        elif entry["status"] == "backlog":
            out["backlog"] += 1
            if fresh is not None:
                fresh["review"] = {"key": key, "status": "backlog", "comment": entry.get("comment") or "", "at": entry.get("at"),
                                   "round": entry.get("round"),
                                   "changed": not drift_of(entry, fresh)["same"]}
        elif entry["status"] == "rejected":
            out["rejected"] += 1
            if fresh is not None:
                taken.add(records.index(fresh))
                replaced[records.index(fresh)] = None
    kept: list[dict] = []
    for i, rec in enumerate(records):
        if i in replaced:
            if replaced[i] is not None:
                kept.append(replaced[i])
        else:
            kept.append(rec)
    kept.extend(extra)
    out["records"] = kept
    return out


def check_book(book: str, records: list[dict], review_dir: str | None = None) -> dict:
    """
    The regression gate for one book: every accepted entry against the
    current records (the fresh cut, before the ledger was applied): same
    shabad, same pages, extent IoU >= IOU_PASS -> pass. Backlog entries
    whose fresh record changed since the comment are listed to re-present.
    """
    ledger = read_ledger(book, review_dir)
    assign_keys(records)
    passed, failed, changed, lost = [], [], [], []
    attached = attach(ledger, records)
    by_entry = {id(e): nid for nid, e in attached.items()}
    by_nid = {r["notation_id"]: r for r in records}
    for key, entry in sorted(ledger.items()):
        fresh = by_nid.get(by_entry.get(id(entry)))
        d = drift_of(entry, fresh)
        if entry["status"] == "accepted":
            if fresh is None:
                lost.append(d)
            elif d["same"]:
                passed.append(key)
            else:
                failed.append(d)
        elif entry["status"] == "backlog" and fresh is not None and not d["same"]:
            changed.append({**d, "comment": entry.get("comment") or ""})
    return {"book": book, "accepted": sum(1 for e in ledger.values() if e["status"] == "accepted"),
            "passed": len(passed), "failed": failed, "lost": lost, "backlog_changed": changed,
            "ok": not failed and not lost}


def status_of(book: str, records: list[dict], review_dir: str | None = None) -> dict:
    """Counts for the dashboard: notations, accepted, backlog, rejected, unreviewed, and whether the book is clear."""
    ledger = read_ledger(book, review_dir)
    assign_keys(records)
    by_status = {s: 0 for s in STATUSES}
    for e in ledger.values():
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    attached = attach(ledger, records)
    unreviewed = sum(1 for r in records if r["notation_id"] not in attached)
    return {"book": book, "notations": len(records), **by_status, "unreviewed": unreviewed,
            "clear": unreviewed == 0 and by_status["backlog"] == 0 and len(records) > 0}
