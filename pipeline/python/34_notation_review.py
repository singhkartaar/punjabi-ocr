"""
The review of a book's notations: tick, comment, reject -- saved as you go.

  34_notation_review.py serve  --book gurmat-sangeet-sagar-1 [--port 8777] [--all]
  34_notation_review.py serve  --sample _firstcut [--port 8777]          # a sampler run's windows, every book in it
  34_notation_review.py check  --book gurmat-sangeet-sagar-1 [--strict]  # the regression gate on what was accepted
  34_notation_review.py status [--book KEY]                              # the dashboard, review/index.html
  34_notation_review.py apply  --book KEY notation-comments.jsonl [--round N]   # a downloaded verdict file into the ledger

`serve` opens a local page of the book's notations that are not yet
accepted -- each as printed, with its page and the page on either side,
its review key, and the earlier comment when it is in the backlog -- and
three buttons a card: Accept (the cut and the shabad link are right),
To backlog (the comment says what is wrong), Not a notation. A verdict is
appended to the ledger (lib/notation_review.py, REVIEW_DIR/<book>.jsonl)
the moment it is given; an accepted notation's record and images are
saved as a fixture, and later runs of 29_notation_parse.py emit that
fixture in its place. Keys: `a` accept, `b` backlog (focuses the
comment), `x` reject, `j`/`k` next and previous card.

`check` is what runs after every change to the reader: each accepted
notation's fixture against the current cut of the same pages -- same
shabad, same pages, extent overlap of 0.9 or more -- and the backlog
entries whose cut changed since the comment (to be shown again).
`--strict` exits 1 when anything accepted fails or is lost.
"""
from __future__ import annotations
import argparse
import html as html_mod
import importlib.util
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib.notation import read_jsonl  # noqa: E402
from lib.notation_review import (STATUSES, append_entry, assign_keys, attach, check_book, entry_of, match_entry,  # noqa: E402
                                 read_ledger, save_fixture, status_of)
from lib.paths import NOTATIONS_DIR, OCR_DIR, REVIEW_DIR  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), os.path.join(HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fresh_records(book: str) -> tuple[dict, list[dict]]:
    """The book's records as 29 wrote them (the ledger already applied there: accepted ones carry `review`)."""
    path = os.path.join(NOTATIONS_DIR, book, "notations.jsonl")
    if not os.path.exists(path):
        return {}, []
    meta, records = read_jsonl(path)
    assign_keys(records)
    return meta, records


def page_files(book: str) -> dict[int, str]:
    p = os.path.join(OCR_DIR, book, "pages.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return {r["page"]: r["file"] for r in json.load(fh).get("pages", [])}


# ---- the page --------------------------------------------------------------

REVIEW_CSS = """
.verdict{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
.verdict button{font:inherit;padding:6px 12px;border:1px solid #999;border-radius:4px;background:#fff;cursor:pointer}
.verdict button.accept{border-color:#2e7d32;color:#2e7d32}.verdict button.backlog{border-color:#b26a00;color:#b26a00}
.verdict button.reject{border-color:#999;color:#666}
.verdict .state{margin-left:auto;font-size:12px;color:#555}
section.card.done-accepted{opacity:.45;border-left:6px solid #2e7d32}section.card.done-backlog{border-left:6px solid #e0a000}
section.card.done-rejected{opacity:.35;border-left:6px solid #999}
.rkey{font-family:ui-monospace,monospace;font-size:12px;color:#666;margin:4px 0}
.prior{background:#fff4d6;border:1px solid #e0c060;padding:6px 8px;border-radius:4px;margin:6px 0;font-size:13px}
.prior b{color:#8a5a00}
.flag-long{background:#fde2e2;color:#a00;padding:1px 6px;border-radius:3px;font-size:12px;margin-left:6px}
.pager{display:flex;gap:8px;margin:12px 16px;align-items:center}.pager a{padding:4px 8px;border:1px solid #ccc;border-radius:4px;color:#333;text-decoration:none}
.pager a.cur{background:#333;color:#fff}.pager a.done{color:#999}.pager a small{font-size:11px}
section.card.stub{display:block;opacity:.6}section.card.stub h2{border:0}
section.card.focus{outline:3px solid #4a90d9}
"""

REVIEW_JS = r"""
const DATA = JSON.parse(document.getElementById('cands').textContent);
const KEYS = DATA.keys;
function card(id) { return document.getElementById('c-' + id); }   // a literal id, not a selector
function setState(id, status, comment) {
  const el = card(id); if (!el) return;
  el.classList.remove('done-accepted', 'done-backlog', 'done-rejected');
  if (status) el.classList.add('done-' + status);
  const st = el.querySelector('.verdict .state'); if (st) st.textContent = status ? (status + (comment ? ' · ' + comment : '')) : '';
}
async function verdict(id, status) {
  const el = card(id); if (!el || el.classList.contains('stub')) return;
  const ta = el.querySelector('textarea.comment'); const comment = ta ? ta.value.trim() : '';
  if (status === 'backlog' && !comment) { ta.focus(); ta.placeholder = 'say what is wrong, then press To backlog again'; return; }
  const r = await fetch('/verdict', {method: 'POST', headers: {'content-type': 'application/json'},
                                     body: JSON.stringify({notation_id: id, key: KEYS[id], status, comment})});
  const got = await r.json();
  if (!r.ok) { alert(got.error || 'not saved'); return; }
  setState(id, status, comment);
  document.getElementById('count').textContent = got.counts;
}
let focused = 0;
const cards = () => [...document.querySelectorAll('section.card:not(.stub)')];
function focus(i) {
  const cs = cards(); if (!cs.length) return;
  focused = Math.max(0, Math.min(cs.length - 1, i));
  cs.forEach(c => c.classList.remove('focus')); cs[focused].classList.add('focus');
  cs[focused].scrollIntoView({block: 'start', behavior: 'smooth'});
}
document.addEventListener('click', e => {
  const t = e.target;
  if (t.tagName === 'IMG' && t.closest('.crops')) t.classList.toggle('zoom');
  if (t.dataset.verdict) verdict(t.dataset.vid, t.dataset.verdict);
  const c = t.closest('section.card'); if (c) focused = cards().indexOf(c);
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  const cs = cards(); if (!cs.length) return;
  const id = cs[focused].id.slice(2);
  if (e.key === 'j') focus(focused + 1);
  else if (e.key === 'k') focus(focused - 1);
  else if (e.key === 'a') verdict(id, 'accepted').then(() => focus(focused + 1));
  else if (e.key === 'x') verdict(id, 'rejected').then(() => focus(focused + 1));
  else if (e.key === 'b') { const ta = cs[focused].querySelector('textarea.comment'); if (ta) ta.focus(); }
});
for (const [id, s] of Object.entries(DATA.states)) if (!s.changed) setState(id, s.status, s.comment);
"""


def build_page(books: list[dict], states: dict[str, dict], counts: str, page_no: int, n_pages: int, title: str, all_: bool,
               left: list[int] | None = None) -> str:
    """
    One review page: `books` is [{"book", "title", "author", "records", "img_base", "pages_base", "files"}];
    `states` is {notation_id: {"status", "comment"}} for the verdicts already given; `left` how many
    cards each page still has to review (the pager prints it).
    """
    gt = load_script("30_notation_gt.py")
    parts: list[str] = []
    keys: dict[str, str] = {}
    cands: list[dict] = []
    for b in books:
        parts.append('<h2 class="book" id="b-%s">%s <small>%s · %d notation(s) here</small></h2>'
                     % (html_mod.escape(b["book"]), html_mod.escape(b["title"]), html_mod.escape(b["author"]), len(b["records"])))
        for rec in b["records"]:
            cand = gt.candidate(rec)
            cand["book"] = b["book"]
            cands.append(cand)
            keys[rec["notation_id"]] = rec["review_key"]
            st = states.get(rec["notation_id"]) or {}
            if not all_ and st.get("status") in ("accepted", "rejected") and not st.get("changed"):   # collapsed in its place: the page keeps its shape
                parts.append('<section class="card stub done-%s" id="c-%s"><h2>%s <small>pages %s · %s%s</small></h2></section>'
                             % (st["status"], html_mod.escape(rec["notation_id"]), html_mod.escape(rec["review_key"]),
                                "-".join(str(p) for p in rec.get("pages") or []), st["status"],
                                " · " + html_mod.escape(st.get("comment") or "") if st.get("comment") else ""))
                continue
            html = gt.card(rec, cand, b["img_base"], {}, b["pages_base"], b["files"], comments=True)
            prior = rec.get("review") or {}
            extra: list[str] = []           # the card already prints the key, the long-span flag and the prior verdict
            verdict = ('<div class="verdict"><button class="accept" data-verdict="accepted" data-vid="%s">Accept (a)</button>'
                       '<button class="backlog" data-verdict="backlog" data-vid="%s">To backlog (b)</button>'
                       '<button class="reject" data-verdict="rejected" data-vid="%s">Not a notation (x)</button>'
                       '<span class="state"></span></div>' % ((html_mod.escape(rec["notation_id"]),) * 3))
            html = html.replace("</div></section>", verdict + "</div></section>")     # the verdict row closes the card
            parts.append(html)
    pager = ""
    if n_pages > 1:
        def link(i):
            n = left[i - 1] if left and i - 1 < len(left) else None
            return '<a href="/?page=%d%s" class="%s%s">%d%s</a>' % (i, "&all=1" if all_ else "", "cur" if i == page_no else "",
                                                                    " done" if n == 0 else "", i, "" if n is None else " <small>(%d left)</small>" % n)
        pager = '<div class="pager">page ' + " ".join(link(i) for i in range(1, n_pages + 1)) + "</div>"
    data = json.dumps({"book": title, "mode": "comments", "candidates": cands, "keys": keys, "states": states},
                      ensure_ascii=False).replace("</", "<\\/")
    return ("<!doctype html><html lang=en><meta charset=utf-8><title>%s · review</title>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<style>%s%s%s</style><body>"
            "<header><h1>%s <small>%s</small></h1><span id=count class=count>%s</span></header>%s%s%s"
            "<script id=cands type=application/json>%s</script><script>%s</script></html>"
            % (html_mod.escape(title), gt.PAGE_CSS, gt.NOTATION_CSS, REVIEW_CSS, html_mod.escape(title),
               "accepted collapsed in place; a verdict is saved the moment it is given" if not all_ else "every notation, accepted ones greyed",
               html_mod.escape(counts), pager, "".join(parts), pager, data, REVIEW_JS))


# ---- the server ------------------------------------------------------------

class ReviewState:
    """What the server shows: the books, their records, and the ledger verdicts; locked for writes."""

    def __init__(self, books: list[dict], per_page: int, all_: bool, round_: int | None, review_dir: str | None, title: str):
        self.books, self.per_page, self.all, self.round, self.review_dir, self.title = books, per_page, all_, round_, review_dir, title
        self.lock = threading.Lock()
        self.by_id: dict[str, tuple[dict, dict]] = {}
        for b in books:
            for rec in b["records"]:
                self.by_id[rec["notation_id"]] = (rec, b)

    def states(self) -> dict[str, dict]:
        out = {}
        for b in self.books:
            ledger = read_ledger(b["book"], self.review_dir)
            by_nid = {r["notation_id"]: r for r in b["records"]}
            for nid, e in attach(ledger, b["records"]).items():       # keys first, then overlap; no record twice
                rv = by_nid[nid].get("review") or {}
                out[nid] = {"status": e["status"], "comment": e.get("comment") or "",
                            "changed": bool(rv.get("noted") and rv.get("changed"))}    # accepted with a note, and the cut moved since
        return out

    def counts(self) -> str:
        st = self.states()
        n = sum(len(b["records"]) for b in self.books)
        acc = sum(1 for s in st.values() if s["status"] == "accepted")
        noted = sum(1 for s in st.values() if s["status"] == "accepted" and s.get("comment"))
        bl = sum(1 for s in st.values() if s["status"] == "backlog")
        rj = sum(1 for s in st.values() if s["status"] == "rejected")
        again = sum(1 for s in st.values() if s.get("changed"))
        return "%d notation(s) · %d accepted%s · %d in backlog · %d not notations · %d to review%s" % (
            n, acc, " (%d with a note)" % noted if noted else "", bl, rj, n - acc - bl - rj,
            " · %d moved under a note: look again" % again if again else "")

    def page(self, page_no: int) -> str:
        """
        Page N is always the same N cards: the pages are cut over every
        record in its fixed place, and a card that is accepted or rejected
        collapses to a one-line stub there (shown whole with --all). A
        page cut over the unreviewed ones alone shifted under the
        reviewer: accepting page 1 moved page 2's cards onto page 1.
        """
        st = self.states()
        flat = [(b, r) for b in self.books for r in b["records"]]
        n_pages = max(1, (len(flat) + self.per_page - 1) // self.per_page)
        page_no = max(1, min(n_pages, page_no))
        left = []                                    # what each page still has to review
        for i in range(n_pages):
            chunk = flat[i * self.per_page: (i + 1) * self.per_page]
            left.append(sum(1 for _, r in chunk if st.get(r["notation_id"], {}).get("status") not in ("accepted", "rejected")
                            or st.get(r["notation_id"], {}).get("changed")))
        chunk = flat[(page_no - 1) * self.per_page: page_no * self.per_page]
        shown: list[dict] = []
        for b, r in chunk:
            if not shown or shown[-1]["book"] != b["book"]:
                shown.append({**b, "records": []})
            shown[-1]["records"].append(r)
        return build_page(shown, st, self.counts(), page_no, n_pages, self.title, self.all, left)

    def verdict(self, notation_id: str, status: str, comment: str) -> dict:
        if status not in STATUSES:
            raise ValueError("status must be one of %s" % ", ".join(STATUSES))
        if notation_id not in self.by_id:
            raise KeyError(notation_id)
        rec, b = self.by_id[notation_id]
        with self.lock:
            entry = entry_of(rec, status, comment, round_=self.round)
            append_entry(entry, self.review_dir)
            if status == "accepted":
                save_fixture(rec, entry, os.path.join(NOTATIONS_DIR, b["book"], "images"), self.review_dir)
        return {"ok": True, "key": entry["key"], "counts": self.counts()}


def make_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):      # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/":
                q = urllib.parse.parse_qs(url.query)
                page_no = int((q.get("page") or ["1"])[0])
                if q.get("all"):
                    state.all = True
                return self._send(200, state.page(page_no).encode("utf-8"))
            if url.path == "/counts":
                return self._send(200, json.dumps({"counts": state.counts()}).encode("utf-8"), "application/json")
            parts = url.path.split("/")
            if len(parts) >= 5 and parts[1] == "b" and parts[3] in ("pages", "images"):
                book, kind, name = parts[2], parts[3], urllib.parse.unquote("/".join(parts[4:]))
                base = os.path.join(OCR_DIR, book, "pages") if kind == "pages" else os.path.join(NOTATIONS_DIR, book, "images")
                path = os.path.normpath(os.path.join(base, name))
                if not path.startswith(os.path.normpath(base) + os.sep) or not os.path.isfile(path):
                    return self._send(404, b"not found", "text/plain")
                with open(path, "rb") as fh:
                    return self._send(200, fh.read(), "image/png")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/verdict":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("content-length") or 0)
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                got = state.verdict(body["notation_id"], body["status"], body.get("comment") or "")
            except (KeyError, ValueError, json.JSONDecodeError) as err:
                return self._send(400, json.dumps({"error": str(err)}).encode("utf-8"), "application/json")
            return self._send(200, json.dumps(got, ensure_ascii=False).encode("utf-8"), "application/json")
    return Handler


def book_entry(book: str, records: list[dict], meta: dict) -> dict:
    return {"book": book, "title": meta.get("title") or book, "author": meta.get("author") or "", "records": records,
            "img_base": "/b/%s" % book, "pages_base": "/b/%s/pages" % book, "files": page_files(book)}


def books_for_sample(name: str) -> list[dict]:
    """The books and windows of a sampler run (OCR_DIR/<name>/sample.json): the notations that begin in its windows."""
    path = os.path.join(OCR_DIR, name, "sample.json")
    if not os.path.exists(path):
        sys.exit("no %s" % path)
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    out = []
    for b in plan:
        meta, records = fresh_records(b["book"])
        wanted = {p for a, c in b["windows"] for p in range(a, c + 1)}
        recs = [r for r in records if r.get("pages") and r["pages"][0] in wanted]
        if recs:
            out.append(book_entry(b["book"], recs, {"title": b.get("title"), "author": b.get("author")}))
    return out


def serve(args) -> None:
    if args.sample:
        books = books_for_sample(args.sample)
        title = args.sample
    else:
        meta, records = fresh_records(args.book)
        if not records:
            sys.exit("no notations for %s; run 29_notation_parse.py first" % args.book)
        books = [book_entry(args.book, records, meta)]
        title = meta.get("title") or args.book
    state = ReviewState(books, args.per_page, args.all, args.round, args.review_dir, title)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print("review: http://127.0.0.1:%d/  (%s)  ledger %s" % (args.port, state.counts(), args.review_dir or REVIEW_DIR))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def check(args) -> None:
    books = [args.book] if args.book else sorted(f[:-6] for f in os.listdir(args.review_dir or REVIEW_DIR) if f.endswith(".jsonl"))
    bad = 0
    for book in books:
        _, records = fresh_records(book)
        # the fresh cut: the records 29 wrote minus the frozen ones (their drift is what 29 reported); the
        # fixtures are compared with the drift the parser saw, kept in the report
        report = None
        rp = os.path.join(HERE, "..", "..", "data", "raw", "notation-report-%s.json" % book)
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as fh:
                report = json.load(fh).get("review")
        # the records as written hold the frozen fixtures in place of the fresh cut; the parser's
        # report keeps the fresh cut's drift, which is the finer witness when it exists
        got = check_book(book, records, args.review_dir)
        if report and report.get("drift"):
            # the parser's own drift (fresh cut against the fixture) is the finer witness when it exists
            notes = {k: e.get("comment") or "" for k, e in read_ledger(book, args.review_dir).items() if e["status"] == "accepted"}
            moved = [d for d in report["drift"] if not d.get("lost") and not d.get("same")]
            failed = [d for d in moved if not notes.get(d["key"])]
            # under a note any move counts, however small (the fresh cut is shown again)
            noted = [{**d, "comment": notes[d["key"]]} for d in report["drift"]
                     if notes.get(d["key"]) and not d.get("lost") and not d.get("identical", d.get("same"))]
            lost = [d for d in report["drift"] if d.get("lost")]
            got.update({"passed": len(report["drift"]) - len(moved) - len(lost), "failed": failed, "lost": lost, "noted": noted,
                        "ok": not failed and not lost})
        print("%-50s accepted %3d  passed %3d  failed %2d  lost %2d  noted moved %2d  backlog changed %2d  %s"
              % (book[:50], got["accepted"], got["passed"], len(got["failed"]), len(got["lost"]), len(got.get("noted", [])),
                 len(got["backlog_changed"]), "ok" if got["ok"] else "FAIL"))
        for d in got["failed"]:
            print("   moved  %s: pages %s -> %s, shabad %s -> %s, overlap %.2f" % (d["key"], d["pages_before"], d.get("pages_after"), d["shabad_before"], d.get("shabad_after"), d.get("extent_iou", 0)))
        for d in got["lost"]:
            print("   lost   %s: pages %s" % (d["key"], d["pages_before"]))
        for d in got.get("noted", []):
            print("   noted, moved  %s: pages %s -> %s (%s)" % (d["key"], d["pages_before"], d.get("pages_after"), d["comment"][:60]))
        for d in got["backlog_changed"]:
            print("   changed, show again  %s: %s" % (d["key"], d["comment"][:60]))
        bad += 0 if got["ok"] else 1
    if args.strict and bad:
        sys.exit(1)


def status(args) -> None:
    review_dir = args.review_dir or REVIEW_DIR
    books = [args.book] if args.book else sorted(f[:-6] for f in os.listdir(review_dir) if f.endswith(".jsonl")) if os.path.isdir(review_dir) else []
    rows = []
    for book in books:
        _, records = fresh_records(book)
        rows.append(status_of(book, records, review_dir))
        r = rows[-1]
        print("%-50s notations %4d  accepted %4d (%d noted)  backlog %3d  rejected %3d  unreviewed %4d  %s"
              % (book[:50], r["notations"], r["accepted"], r["noted"], r["backlog"], r["rejected"], r["unreviewed"], "CLEAR" if r["clear"] else ""))
    os.makedirs(review_dir, exist_ok=True)
    with open(os.path.join(review_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset=utf-8><title>notation review</title>"
                 "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:right}"
                 "td:first-child,th:first-child{text-align:left}tr.clear{background:#e3f4e6}</style>"
                 "<h1>Notation review</h1><table><tr><th>book</th><th>notations</th><th>accepted</th><th>with a note</th><th>backlog</th><th>not notations</th><th>unreviewed</th></tr>"
                 + "".join("<tr class='%s'><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td></tr>"
                           % ("clear" if r["clear"] else "", html_mod.escape(r["book"]), r["notations"], r["accepted"], r["noted"], r["backlog"], r["rejected"], r["unreviewed"])
                           for r in rows)
                 + "</table><p>Serve a book with <code>34_notation_review.py serve --book &lt;key&gt;</code>.</p>")
    print("-> %s" % os.path.join(review_dir, "index.html"))


def apply(args) -> None:
    """A downloaded verdict file (notation-comments.jsonl, or 30's candidates) into the ledger, matched by id then by overlap."""
    _, records = fresh_records(args.book)
    by_id = {r["notation_id"]: r for r in records}
    images_dir = os.path.join(NOTATIONS_DIR, args.book, "images")
    n = 0
    with open(args.file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            v = json.loads(line)
            rec = by_id.get(v.get("notation_id"))
            if rec is None and v.get("pages"):
                probe = {"book_key": args.book, "shabad_id": v.get("shabad_id"), "pages": v["pages"], "extent": {}, "nth": None}
                rec = match_entry(probe, records) if False else None       # no extent in a download: id only
            if rec is None:
                continue
            comment = (v.get("comment") or v.get("note") or "").strip()
            if comment:
                status_ = "backlog"
            elif v.get("looks_right") or v.get("verified"):
                status_ = "accepted"
            else:
                continue
            entry = entry_of(rec, status_, comment, round_=args.round)
            append_entry(entry, args.review_dir)
            if status_ == "accepted":
                save_fixture(rec, entry, images_dir, args.review_dir)
            n += 1
    print("%d verdict(s) into %s" % (n, os.path.join(args.review_dir or REVIEW_DIR, args.book + ".jsonl")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--book"); s.add_argument("--sample"); s.add_argument("--port", type=int, default=8777)
    s.add_argument("--all", action="store_true"); s.add_argument("--per-page", type=int, default=40); s.add_argument("--round", type=int)
    s.add_argument("--review-dir")
    c = sub.add_parser("check"); c.add_argument("--book"); c.add_argument("--strict", action="store_true"); c.add_argument("--review-dir")
    t = sub.add_parser("status"); t.add_argument("--book"); t.add_argument("--review-dir")
    a = sub.add_parser("apply"); a.add_argument("--book", required=True); a.add_argument("file"); a.add_argument("--round", type=int); a.add_argument("--review-dir")
    args = ap.parse_args()
    if args.cmd == "serve":
        if not args.book and not args.sample:
            sys.exit("serve needs --book or --sample")
        serve(args)
    elif args.cmd == "check":
        check(args)
    elif args.cmd == "status":
        status(args)
    elif args.cmd == "apply":
        apply(args)


if __name__ == "__main__":
    main()
