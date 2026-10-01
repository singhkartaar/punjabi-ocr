"""
Ground truth for a notation book: the review window, the review page, the gold.

  30_notation_gt.py --book gurmat-sangeet-sagar-1 --mid           # the protocol's window
  30_notation_gt.py --book gurmat-sangeet-sagar-1 --pages 165-176 # an explicit window
  30_notation_gt.py --book gurmat-sangeet-sagar-1 --all
  30_notation_gt.py --book gurmat-sangeet-sagar-1 --check
  30_notation_gt.py --book gurmat-sangeet-sagar-1 --promote

Runs after 29_notation_parse.py. --mid is the review protocol (docs/notations.md):
five consecutive pages from about 45% of the book, extended to seven, and
past seven a page at a time, until at least two notations with a resolved
shabad are covered. The chosen records go to data/ocr/<book>/gt/
notation-candidates.jsonl, one per notation, with every judged field
prefilled from the record, and gt/notation-review.html shows each notation's
crops beside what the parser made of them: the shabad it named (first line,
ang, writer, the corpus raag), the heading it read (raag used, taal, laya),
the sections it found, and -- once the grid reader runs -- the grid as the
shared rendering, in Gurmukhi and in English.

The page is the reviewer's tool: mark each field right or wrong, type the
correction, mark the notation verified, and download the edited
candidates.jsonl over the one on disk. --check validates the edits;
--promote appends the verified candidates to gt/notation-gold.jsonl
(replacing an earlier record with the same id), which 31_notation_eval.py
measures and 32_build_notations_db.py applies.
"""
from __future__ import annotations
import argparse
import datetime as dt
import html as html_mod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import notation
from lib.notation import mid_window, read_jsonl
from lib.notation_render import NOTATION_CSS
from lib.notation_render import html as render_html
from lib.notation_vocab import LAYA, RAAGS, TAALS, VOCAB_PATH
from lib.ocr_pages import parse_pages
from lib.paths import NOTATIONS_DIR, OCR_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WINDOW, WINDOW_MAX, MIN_RESOLVED = 5, 7, 2
FIELDS = ("shabad", "raag_used", "taal", "laya", "structure")
STATUSES = ("ok", "partial", "skip")


def load_records(book: str) -> tuple[dict, list[dict], str]:
    path = os.path.join(NOTATIONS_DIR, book, "notations.jsonl")
    if not os.path.exists(path):
        sys.exit("no %s; run 29_notation_parse.py --book %s first" % (path, book))
    meta, records = read_jsonl(path)
    return meta, records, os.path.dirname(path)


def book_pages(book: str) -> int:
    p = os.path.join(OCR_DIR, book, "pages.json")
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as fh:
        meta = json.load(fh)
    return int(((meta.get("probe") or {}).get("pages")) or len(meta.get("pages") or []))


def resolved(rec: dict) -> bool:
    return (rec.get("shabad") or {}).get("shabad_id") is not None


def choose_window(records: list[dict], n_pages: int) -> tuple[int, int, list[dict], str | None]:
    """The protocol's window over the records that exist; a warning when it could not be met."""
    a, b = mid_window(n_pages or (max(r["page"] for r in records) if records else 1), WINDOW)
    have = sorted({p for r in records for p in r["pages"]})

    def within(lo, hi):
        return [r for r in records if any(lo <= p <= hi for p in r["pages"])]

    while sum(1 for r in within(a, b) if resolved(r)) < MIN_RESOLVED and b - a + 1 < WINDOW_MAX and b < (n_pages or b + 1):
        b += 1
    while sum(1 for r in within(a, b) if resolved(r)) < MIN_RESOLVED and (b + 1) in have:
        b += 1
    chosen = within(a, b)
    warn = None
    got = sum(1 for r in chosen if resolved(r))
    if got < MIN_RESOLVED:
        missing = [p for p in range(a, b + 1) if p not in have]
        warn = ("only %d resolved notation(s) on pages %d-%d; %s"
                % (got, a, b, ("pages %s have no parse yet: run 29_notation_parse.py --pages %d-%d and rerun --mid"
                               % (",".join(map(str, missing)), a, b + 3)) if missing
                   else "extend with --pages %d-%d after parsing more pages" % (a, b + 3)))
    return a, b, chosen, warn


def candidate(rec: dict) -> dict:
    """The record's judged fields, prefilled: the reviewer flips ok and types a correction."""
    h = rec.get("heading") or {}
    sh = rec.get("shabad") or {}
    grid_img = next((im for im in rec.get("images", []) if im["role"] == "grid"), None)
    return {
        "notation_id": rec["notation_id"], "page": rec["page"], "pages": rec["pages"],
        "bbox": grid_img["bbox"] if grid_img else None, "content_hash": rec["source"]["content_hash"],
        "by": "", "at": "", "status": "ok", "verified": False,
        "shabad": {"ok": None, "value": sh.get("shabad_id"), "correct": None},
        "raag_used": {"ok": None, "value": (h.get("raag") or {}).get("key"), "correct": None},
        "taal": {"ok": None, "value": (h.get("taal") or {}).get("key"), "correct": None},
        "laya": {"ok": None, "value": (h.get("taal") or {}).get("laya"), "correct": None},
        "structure": {"ok": None, "value": structure_of(rec), "correct": None},
        "cells": [], "cells_checked": "all", "note": "",
        "draft": {"heading": h.get("raw"), "first_line": sh.get("first_line"), "ang": sh.get("ang"),
                  "confidence": sh.get("confidence"), "kind": rec["kind"], "flags": rec.get("flags", [])},
    }


def structure_of(rec: dict) -> str:
    """'sthai,antara' -- the section kinds in order, from the parsed grid or the page layout."""
    secs = rec.get("sections") or (rec.get("layout") or {}).get("sections") or []
    return ",".join(str(s.get("kind") or "?") for s in secs)


# ---- the review page --------------------------------------------------------

PAGE_CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f5f4f0;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 16px;display:flex;gap:16px;align-items:center;z-index:2}
header h1{font-size:16px;margin:0;flex:1}
button{font:inherit;padding:4px 10px;border:1px solid #888;border-radius:4px;background:#fff;cursor:pointer}
button.primary{background:#2a5db0;color:#fff;border-color:#2a5db0}
.card{background:#fff;margin:14px 16px;border:1px solid #ddd;border-radius:6px;display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:0}
.card.verified{border-color:#3a9a4a;box-shadow:0 0 0 2px #cfe9d3 inset}
.card h2{grid-column:1/-1;font-size:15px;margin:0;padding:8px 12px;border-bottom:1px solid #eee;background:#fafaf7;display:flex;gap:12px;align-items:baseline}
.card h2 small{color:#666;font-weight:normal}
.crops{padding:10px 12px;border-right:1px solid #eee;min-width:0}
.crops img{max-width:100%;height:auto;border:1px solid #ccc;display:block;margin:0 0 8px;background:#fff;cursor:zoom-in}
.crops img.zoom{max-width:none;cursor:zoom-out}
.crops .origs{display:flex;gap:10px;flex-wrap:wrap}.crops .origs figure{margin:0}.crops .origs figcaption{font-size:11px;color:#777}
.crops .origs img{width:220px;height:auto}.crops .origs img.zoom{width:auto;max-width:100%}.crops .origs figure.ctx{opacity:.55;border:1px dashed #bbb;padding:4px;border-radius:4px}.crops .origs figure.ctx:hover{opacity:1}.crops .origs .noimg{width:220px;height:120px;display:flex;align-items:center;justify-content:center;color:#999;font-size:12px;background:#f4f4f4}
.crops details{margin:8px 0}.crops summary,.facts summary{cursor:pointer;color:#555;font-size:13px}
.pages-note{background:#fff4d6;border:1px solid #f0d890;border-radius:4px;padding:4px 8px;font-size:13px;margin:0 0 8px}
.raags{margin:14px 16px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 12px}
.raags h3{margin:0 0 6px;font-size:15px}.raags .note{display:flex;gap:12px;margin:8px 0;align-items:flex-start}
.raags .note img{max-width:60%;height:auto;border:1px solid #ccc;cursor:zoom-in}.raags .note img.zoom{max-width:none;cursor:zoom-out}
.raags .note .txt{font-size:13px;color:#333;white-space:pre-wrap;max-height:16em;overflow:auto}
.crops .role{font-size:11px;color:#777;text-transform:uppercase;letter-spacing:.04em}
.facts{padding:10px 12px;min-width:0}
table.f{border-collapse:collapse;width:100%}
table.f td,table.f th{padding:4px 6px;border-bottom:1px solid #eee;vertical-align:top;text-align:left}
table.f th{width:92px;color:#555;font-weight:600}
.gur{font-family:'Noto Sans Gurmukhi','Raavi',sans-serif;font-size:16px}
.judge{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.judge input[type=text]{font:inherit;padding:2px 4px;border:1px solid #bbb;border-radius:3px;min-width:12em}
.judge label{white-space:nowrap}
.flags span{display:inline-block;background:#eee;border-radius:3px;padding:0 5px;margin:0 3px 3px 0;font-size:12px}
.flags span.bad{background:#fde2e2}
.printed{color:#444;font-size:13px;white-space:pre-wrap}
textarea{width:100%;box-sizing:border-box;font:inherit;min-height:3em}
textarea.comment{min-height:5em;border:2px solid #c9a227;border-radius:4px;padding:6px;background:#fffdf5}
.row.comment{margin-top:12px}
.row{display:flex;gap:10px;align-items:center;margin-top:8px;flex-wrap:wrap}
.count{color:#555}
.ntn{overflow-x:auto;margin-top:8px}
"""

PAGE_JS = r"""
const DATA = JSON.parse(document.getElementById('cands').textContent);
const KEY = 'notation-review:' + DATA.book;
const FIELDS = ['shabad','raag_used','taal','laya','structure'];
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} update(); }
function st(id) { return state[id] || (state[id] = {}); }
function merged(c) {
  const s = state[c.notation_id] || {};
  const out = JSON.parse(JSON.stringify(c));
  for (const f of FIELDS) {
    if (s[f + '_ok'] !== undefined) out[f].ok = s[f + '_ok'];
    if (s[f + '_correct'] !== undefined) out[f].correct = s[f + '_correct'] === '' ? null : s[f + '_correct'];
  }
  if (s.status) out.status = s.status;
  if (s.note !== undefined) out.note = s.note;
  if (s.verified !== undefined) out.verified = s.verified;
  if (out.verified) { out.by = out.by || DATA.reviewer || 'reviewer'; out.at = s.at || new Date().toISOString(); }
  delete out.draft;
  return out;
}
function update() {
  let v = 0;
  for (const c of DATA.candidates) {
    const el = document.getElementById('c-' + CSS.escape(c.notation_id));
    const s = state[c.notation_id] || {};
    if (!el) continue;
    el.classList.toggle('verified', !!s.verified);
    if (s.verified) v++;
    for (const f of FIELDS) {
      const ok = s[f + '_ok'];
      el.querySelectorAll('input[name="' + c.notation_id + ':' + f + '"]').forEach(r => { r.checked = (r.value === String(ok)); });
      const t = el.querySelector('input[data-correct="' + f + '"]');
      if (t) { t.value = s[f + '_correct'] || ''; t.style.display = ok === false ? '' : 'none'; }
    }
    const ver = el.querySelector('input[data-verified]'); if (ver) ver.checked = !!s.verified;
    const sel = el.querySelector('select[data-status]'); if (sel) sel.value = s.status || 'ok';
    const note = el.querySelector('textarea[data-note]'); if (note && document.activeElement !== note) note.value = s.note || '';
  }
  if (DATA.mode === 'comments') {
    const n = DATA.candidates.filter(c => ((state[c.notation_id] || {}).note || '').trim()).length;
    document.getElementById('count').textContent = n + ' of ' + DATA.candidates.length + ' commented · ' + v + ' marked right';
  } else {
    document.getElementById('count').textContent = v + ' of ' + DATA.candidates.length + ' verified';
  }
}
document.addEventListener('change', e => {
  const t = e.target;
  const id = t.dataset.id; if (!id) return;
  const s = st(id);
  if (t.dataset.field) s[t.dataset.field + '_ok'] = (t.value === 'true');
  if (t.dataset.correct) s[t.dataset.correct + '_correct'] = t.value;
  if (t.dataset.verified !== undefined) { s.verified = t.checked; s.at = new Date().toISOString(); }
  if (t.dataset.status !== undefined) s.status = t.value;
  if (t.dataset.note !== undefined) s.note = t.value;
  save();
});
document.addEventListener('input', e => {
  const t = e.target; if (!t.dataset.id) return;
  if (t.dataset.note !== undefined) { st(t.dataset.id).note = t.value; try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (x) {} }
  if (t.dataset.correct) { st(t.dataset.id)[t.dataset.correct + '_correct'] = t.value; try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (x) {} }
});
document.addEventListener('click', e => {
  const t = e.target;
  if (t.tagName === 'IMG' && t.closest('.crops')) t.classList.toggle('zoom');
  if (t.dataset.allok) {
    const s = st(t.dataset.allok);
    for (const f of FIELDS) s[f + '_ok'] = true;
    s.verified = true; s.at = new Date().toISOString(); s.status = 'ok';
    save();
  }
});
document.getElementById('download').addEventListener('click', () => {
  let lines, name;
  if (DATA.mode === 'comments') {
    lines = DATA.candidates.map(c => { const s = state[c.notation_id] || {}; return JSON.stringify({
      notation_id: c.notation_id, book: c.notation_id.split(':')[0], pages: c.pages || null,
      comment: s.note || '', looks_right: !!s.verified, at: s.at || null}); });
    name = 'notation-comments.jsonl';
  } else {
    lines = DATA.candidates.map(c => JSON.stringify(merged(c)));
    name = 'notation-candidates.jsonl';
  }
  const blob = new Blob([lines.join('\n') + '\n'], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
});
document.getElementById('reset').addEventListener('click', () => {
  if (confirm('Forget every judgement on this page?')) { state = {}; save(); }
});
update();
"""


def esc(v) -> str:
    return html_mod.escape("" if v is None else str(v))


def judge(cid: str, field: str, value, hint: str) -> str:
    name = esc(cid + ":" + field)
    return ('<div class="judge"><b>%s</b> '
            '<label><input type="radio" name="%s" value="true" data-id="%s" data-field="%s"> right</label> '
            '<label><input type="radio" name="%s" value="false" data-id="%s" data-field="%s"> wrong</label> '
            '<input type="text" data-id="%s" data-correct="%s" placeholder="%s" style="display:none"></div>'
            % (esc(value if value not in (None, "") else "—"), name, esc(cid), esc(field), name, esc(cid), esc(field),
               esc(cid), esc(field), esc(hint)))


def describe_sections(secs: list[dict]) -> str:
    """'sthai 1 (1 grid, markers × 0 2 3); antara 1 (2 grids)'"""
    out = []
    for s in secs:
        grids = len(s.get("grids") or [])
        marks = [" ".join(str(x) if x is not None else "·" for x in m["marks"]) for m in (s.get("markers") or []) if m.get("marks")]
        out.append("%s%s (%d grid%s%s)" % (s.get("kind") or "?", (" %s" % s["n"]) if s.get("n") else "",
                                         grids, "" if grids == 1 else "s", (", markers " + "; ".join(marks)) if marks else ""))
    return "; ".join(out) or "none"


def card(rec: dict, cand: dict, img_base: str, corpus_lines: dict, pages_base: str | None = None, page_files: dict | None = None,
         comments: bool = False) -> str:
    """One notation's card. With `comments`, the field judgements give way to a single comment box and a 'looks right' tick."""
    page_files = page_files or {}
    cid = rec["notation_id"]
    h = rec.get("heading") or {}
    sh = rec.get("shabad") or {}
    raag = h.get("raag") or {}
    taal = h.get("taal") or {}
    parts = ['<section class="card" id="c-%s">' % esc(cid)]
    parts.append('<h2>%s <small>pages %s · %s · seq %d</small><span style="flex:1"></span>%s</h2>'
                 % (esc(cid), esc(",".join(map(str, rec["pages"]))), esc(rec["kind"]), rec["seq"],
                    "" if comments else '<button data-allok="%s">all right, verified</button>' % esc(cid)))
    parts.append('<div class="crops">')
    blocks = [im for im in rec.get("images", []) if im["role"] == "block"]
    others = [im for im in rec.get("images", []) if im["role"] != "block"]
    if len(rec["pages"]) > 1:
        parts.append('<div class="pages-note">runs over pages %s: each page is shown whole below, in order</div>' % esc(", ".join(map(str, rec["pages"]))))
    for im in blocks:
        parts.append('<div class="role">as printed · page %d</div><img src="%s" loading="lazy" alt="notation, page %d">'
                     % (im["page"], esc(img_base + "/" + im["file"]), im["page"]))
    if pages_base:
        parts.append('<div class="role">original page%s, with the page before and after</div><div class="origs">' % ("s" if len(rec["pages"]) > 1 else ""))
        before, after = rec["pages"][0] - 1, rec["pages"][-1] + 1
        for p in [before] + list(rec["pages"]) + [after]:
            f = page_files.get(p)
            ctx = p in (before, after)
            if f:
                parts.append('<figure%s><figcaption>page %d%s</figcaption><img src="%s" loading="lazy" alt="page %d"></figure>'
                             % (' class="ctx"' if ctx else "", p, " · before" if p == before else (" · after" if p == after else ""),
                                esc(pages_base + "/" + f), p))
            elif ctx:
                parts.append('<figure class="ctx none"><figcaption>page %d%s</figcaption><div class="noimg">not rendered</div></figure>'
                             % (p, " · before" if p == before else " · after"))
        parts.append("</div>")
    if others:
        parts.append('<details><summary>the parser\'s finer cuts (%d)</summary>' % len(others))
        for im in others:
            parts.append('<div class="role">%s · p.%d</div><img src="%s" loading="lazy" alt="%s">'
                         % (esc(im["role"]), im["page"], esc(img_base + "/" + im["file"]), esc(im["role"])))
        parts.append("</details>")
    if not rec.get("images"):
        parts.append("<p><i>no crops</i></p>")
    parts.append("</div>")
    parts.append('<div class="facts"><table class="f">')
    parts.append("<tr><th>heading</th><td class=gur>%s</td></tr>" % esc(h.get("raw")))
    parts.append("<tr><th>raag used</th><td>%s → <b>%s</b>%s%s</td></tr>"
                 % (esc(raag.get("printed")), esc(raag.get("key")),
                    (" (parent %s)" % esc(raag.get("parent_key"))) if raag.get("parent_key") else "",
                    (" · conf %.2f" % raag["confidence"]) if raag.get("confidence") is not None else ""))
    parts.append("<tr><th>taal</th><td>%s → <b>%s</b>%s%s</td></tr>"
                 % (esc(taal.get("printed")), esc(taal.get("key")),
                    (" · %s matras" % taal["matras"]) if taal.get("matras") else "",
                    (" · laya %s" % esc(taal.get("laya"))) if taal.get("laya") else ""))
    parts.append("<tr><th>shabad</th><td>%s</td></tr>"
                 % ("<b>#%s</b> · ang %s · %s<br><span class=gur>%s</span><br>corpus raag %s%s<br>"
                    "<small>confidence %.2f · %s</small>"
                    % (esc(sh.get("shabad_id")), esc(sh.get("ang")), esc(sh.get("writer")), esc(sh.get("first_line")),
                       esc(rec.get("raag_shabad")), " · <b>differs</b>" if rec.get("raag_differs") else "",
                       sh.get("confidence") or 0.0, esc(sh.get("method")))
                    if sh.get("shabad_id") is not None else
                    "<i>none</i> · %s%s" % (esc(sh.get("method")),
                                             (" · printed ref: %s" % esc((sh.get("ref") or {}).get("text"))) if sh.get("ref") else "")))
    if sh.get("printed"):
        parts.append("<tr><th>text read</th><td class='printed gur'>%s</td></tr>" % esc("\n".join(sh["printed"][:8])))
    secs = (rec.get("layout") or {}).get("sections") or []
    parts.append("<tr><th>sections</th><td>%s</td></tr>" % esc(describe_sections(secs)))
    notes = (rec.get("layout") or {}).get("notes") or []
    if notes:
        parts.append("<tr><th>notes</th><td class=gur>%s</td></tr>" % esc(" / ".join(notes)))
    parts.append('<tr><th>flags</th><td class=flags>%s</td></tr>'
                 % "".join('<span class="%s">%s</span>' % ("bad" if f in ("unresolved-shabad", "ref-conflict", "weak-shabad") else "", esc(f))
                           for f in rec.get("flags", [])))
    parts.append("</table>")
    if rec.get("sections"):
        parts.append('<details class="grids"><summary>the grid as the machine read it (not under review yet)</summary>')
        parts.append('<div class="ntn-wrap">%s</div>' % render_html(rec, "gurmukhi", corpus_lines))
        parts.append('<div class="ntn-wrap">%s</div>' % render_html(rec, "english", corpus_lines))
        parts.append('</details>')
    if comments:
        parts.append('<div class="row comment"><b>your comment</b> <label><input type="checkbox" data-id="%s" data-verified> looks right</label></div>'
                     '<textarea data-id="%s" data-note class="comment" placeholder="what is wrong, missing or cut off here: the shabad, the pages, '
                     'the start or the end of the cut, a taan or a note left out, anything the page shows that the record does not"></textarea>'
                     % (esc(cid), esc(cid)))
        parts.append("</div></section>")
        return "".join(parts)
    parts.append(judge(cid, "shabad", sh.get("shabad_id"), "correct shabad_id, or empty for none"))
    parts.append(judge(cid, "raag_used", raag.get("key"), "raag key, e.g. bhairavi"))
    parts.append(judge(cid, "taal", taal.get("key"), "taal key, e.g. teentaal"))
    parts.append(judge(cid, "laya", taal.get("laya"), "vilambit | madhya | drut"))
    parts.append(judge(cid, "structure", cand["structure"]["value"], "e.g. sthai,antara,antara"))
    parts.append('<div class="row"><label>status <select data-id="%s" data-status>%s</select></label>'
                 '<label><input type="checkbox" data-id="%s" data-verified> verified</label></div>'
                 % (esc(cid), "".join('<option value="%s">%s</option>' % (s, s) for s in STATUSES), esc(cid)))
    parts.append('<textarea data-id="%s" data-note placeholder="note for the parser (what the crop shows that the record does not)"></textarea>' % esc(cid))
    parts.append("</div></section>")
    return "".join(parts)


def raag_notes_html(notes: list[dict], img_base: str, pages: set | None = None) -> str:
    """The book's raag descriptions, cut from the page, each under its raag."""
    shown = [n for n in notes if pages is None or any(p in pages for p in n["pages"])]
    if not shown:
        return ""
    parts = ['<section class="raags"><h3>What the book says about the raags (%d)</h3>' % len(shown)]
    for n in shown:
        raag = n.get("raag") or {}
        parts.append('<div class="note">')
        for im in n.get("images") or ([n["image"]] if n.get("image") else []):
            parts.append('<img src="%s" loading="lazy" alt="raag description">' % esc(img_base + "/" + im["file"]))
        parts.append('<div><b>%s</b> → %s%s · pages %s<div class="txt gur">%s</div></div></div>'
                     % (esc(n.get("heading")), esc(raag.get("key") or "?"),
                        (" (parent %s)" % esc(raag["parent_key"])) if raag.get("parent_key") else "",
                        esc(", ".join(map(str, n["pages"]))), esc((n.get("text") or "")[:1200])))
    parts.append("</section>")
    return "".join(parts)


def load_raag_notes(book_dir: str) -> list[dict]:
    p = os.path.join(book_dir, "raags.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip() and "_meta" not in json.loads(l)]


def page_files_of(book: str) -> dict[int, str]:
    p = os.path.join(OCR_DIR, book, "pages.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return {r["page"]: r["file"] for r in json.load(fh).get("pages", [])}


def review_page(book: str, meta: dict, records: list[dict], cands: list[dict], path: str, img_base: str,
                window: tuple[int, int] | None, warn: str | None, raag_notes: list[dict] | None = None,
                pages_base: str | None = None, page_files: dict | None = None) -> None:
    corpus_lines: dict = {}
    by_id = {r["notation_id"]: r for r in records}
    body = [card(by_id[c["notation_id"]], c, img_base, corpus_lines, pages_base, page_files) for c in cands]
    shown_pages = {p for c in cands for p in (c.get("pages") or [])}
    body.insert(0, raag_notes_html(raag_notes or [], img_base, shown_pages if window else None))
    data = json.dumps({"book": book, "reviewer": os.environ.get("USER") or os.environ.get("USERNAME") or "",
                       "candidates": cands}, ensure_ascii=False).replace("</", "<\\/")
    title = "%s · notation review" % book
    doc = ("<!doctype html><html lang=en><meta charset=utf-8><title>%s</title>"
           "<meta name=viewport content='width=device-width,initial-scale=1'>"
           "<style>%s%s</style><body>"
           "<header><h1>%s <small>%s%s</small></h1><span id=count class=count></span>"
           "<button id=download class=primary>download candidates.jsonl</button><button id=reset>reset</button></header>"
           "%s%s"
           "<script id=cands type=application/json>%s</script><script>%s</script></html>"
           % (esc(title), PAGE_CSS, NOTATION_CSS, esc(meta.get("title") or book),
              (" · pages %d-%d" % window) if window else "", " · %d notations" % len(cands),
              ('<p style="margin:12px 16px;color:#a33">%s</p>' % esc(warn)) if warn else "",
              "".join(body), data, PAGE_JS))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)


# ---- check and promote -------------------------------------------------------

def check_candidates(cands: list[dict]) -> list[str]:
    errs: list[str] = []
    raag_keys = set(RAAGS.keys()) if isinstance(RAAGS, dict) else {r["key"] for r in RAAGS}
    taal_keys = set(TAALS.keys()) if isinstance(TAALS, dict) else {t["key"] for t in TAALS}
    laya_keys = set(LAYA.keys()) if isinstance(LAYA, dict) else {l["key"] for l in LAYA}
    with open(VOCAB_PATH, encoding="utf-8") as fh:
        sec_kinds = set(json.load(fh)["sections"].keys()) | {"other"}
    for c in cands:
        cid = c.get("notation_id", "?")
        if notation.parse_id(cid) is None:
            errs.append("%s: bad id" % cid)
        if c.get("status") not in STATUSES:
            errs.append("%s: status %r" % (cid, c.get("status")))
        if not c.get("verified"):
            continue
        for f in FIELDS:
            j = c.get(f) or {}
            if j.get("ok") not in (True, False):
                errs.append("%s: %s not judged" % (cid, f))
            if j.get("ok") is False:
                v = j.get("correct")
                if f == "shabad" and v not in (None, "") and not str(v).isdigit():
                    errs.append("%s: shabad correction %r is not a shabad_id" % (cid, v))
                if f == "raag_used" and v not in (None, "") and v not in raag_keys:
                    errs.append("%s: raag %r is not a vocabulary key" % (cid, v))
                if f == "taal" and v not in (None, "") and v not in taal_keys:
                    errs.append("%s: taal %r is not a vocabulary key" % (cid, v))
                if f == "laya" and v not in (None, "") and v not in laya_keys:
                    errs.append("%s: laya %r is not one of %s" % (cid, v, sorted(laya_keys)))
                if f == "structure" and v:
                    bad = [k for k in str(v).split(",") if k.strip() and k.strip() not in sec_kinds]
                    if bad:
                        errs.append("%s: structure kinds %s unknown" % (cid, bad))
        for cell in c.get("cells") or []:
            if not all(k in cell for k in ("s", "l", "b", "field", "value")):
                errs.append("%s: cell correction needs s, l, b, field, value" % cid)
            elif cell["field"] not in ("notes", "ext", "rest", "bol", "m", "div", "beats"):
                errs.append("%s: cell field %r" % (cid, cell["field"]))
    return errs


def gold_of(c: dict) -> dict:
    """A verified candidate as the gold record 31 and 32 read."""
    out = {k: v for k, v in c.items() if k != "draft"}
    for f in FIELDS:
        j = dict(out.get(f) or {})
        j["truth"] = j.get("value") if j.get("ok") else (j.get("correct") if j.get("correct") not in ("",) else None)
        if f == "shabad" and j["truth"] not in (None, ""):
            j["truth"] = int(j["truth"])
        out[f] = j
    out["at"] = out.get("at") or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mid", action="store_true", help="the review protocol's mid-book window")
    g.add_argument("--pages", help="an explicit window, e.g. 165-176")
    g.add_argument("--all", action="store_true", help="every notation parsed so far")
    g.add_argument("--check", action="store_true", help="validate gt/notation-candidates.jsonl")
    g.add_argument("--promote", action="store_true", help="verified candidates -> gt/notation-gold.jsonl")
    ap.add_argument("--out", default=OCR_DIR, help="the OCR working directory (default OCR_DIR)")
    args = ap.parse_args()

    gt_dir = os.path.join(args.out, args.book, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    cand_path = os.path.join(gt_dir, "notation-candidates.jsonl")
    gold_path = os.path.join(gt_dir, "notation-gold.jsonl")

    if args.check or args.promote:
        if not os.path.exists(cand_path):
            sys.exit("no %s" % cand_path)
        with open(cand_path, encoding="utf-8") as fh:
            cands = [json.loads(l) for l in fh if l.strip()]
        errs = check_candidates(cands)
        for e in errs:
            print("  " + e)
        verified = [c for c in cands if c.get("verified")]
        print("%d candidates, %d verified, %d problem(s)" % (len(cands), len(verified), len(errs)))
        if errs:
            sys.exit(1)
        if args.promote:
            old: dict[str, dict] = {}
            if os.path.exists(gold_path):
                with open(gold_path, encoding="utf-8") as fh:
                    for l in fh:
                        if l.strip():
                            g_ = json.loads(l)
                            old[g_["notation_id"]] = g_
            for c in verified:
                old[c["notation_id"]] = gold_of(c)
            with open(gold_path, "w", encoding="utf-8", newline="\n") as fh:
                for g_ in old.values():
                    fh.write(json.dumps(g_, ensure_ascii=False) + "\n")
            print("gold: %d record(s) in %s" % (len(old), gold_path))
        return

    meta, records, book_dir = load_records(args.book)
    window, warn = None, None
    if args.pages:
        wanted = {i + 1 for i in parse_pages(args.pages, max(p for r in records for p in r["pages"]))}
        chosen = [r for r in records if any(p in wanted for p in r["pages"])]
        window = (min(wanted), max(wanted))
    elif args.all:
        chosen = list(records)
    else:
        a, b, chosen, warn = choose_window(records, book_pages(args.book))
        window = (a, b)
    if not chosen:
        sys.exit("no notations in the window; parse those pages with 29_notation_parse.py first")
    cands = [candidate(r) for r in chosen]
    with open(cand_path, "w", encoding="utf-8", newline="\n") as fh:
        for c in cands:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    img_base = os.path.relpath(book_dir, gt_dir).replace("\\", "/")
    review = os.path.join(gt_dir, "notation-review.html")
    review_page(args.book, meta, records, cands, review, img_base, window, warn,
                raag_notes=load_raag_notes(book_dir), pages_base="../pages", page_files=page_files_of(args.book))
    n_res = sum(1 for r in chosen if resolved(r))
    print("%s: %d notation(s)%s, %d with a shabad -> %s" % (args.book, len(chosen), (" on pages %d-%d" % window) if window else "",
                                                             n_res, review))
    if warn:
        print("WARNING: " + warn)


if __name__ == "__main__":
    main()
