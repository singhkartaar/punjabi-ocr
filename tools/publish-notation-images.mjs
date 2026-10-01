#!/usr/bin/env node
'use strict';
/**
 * The crops of the notation books -> GitHub release assets in a public
 * repository, and their URLs back into the database and the data.
 *
 *   node tools/publish-notation-images.mjs [--db artifacts/notations.sqlite] [--images <data>/notations]
 *        --repo OWNER/NAME [--book KEY ...] [--all-roles]
 *        [--max N] [--per-release 900] [--batch 25] [--dry-run] [--verify [--sample N]] [--prune]
 *
 * What is published: the images the app shows -- a notation's `block`
 * cuts and their thumbnails, or every image of a notation that has no
 * block (older records) -- unless --all-roles asks for the finer cuts
 * too (grid, shabad, heading: the parser's working material). The
 * database decides which notations: a manual build (32 --accepted-only)
 * publishes the accepted ones, an auto build every one.
 *
 * Where: one series of releases a book, `notations-<book>-v1`, then
 * `notations-<book>-v1-p2`, `-p3` ... when a release holds --per-release
 * assets (GitHub caps a release at 1000). Each image is one asset named
 * after its content -- `<notation id, : as ->-<n>-<kind>-<sha8>.png` --
 * uploaded once per book whatever rows share it, so a rerun uploads only
 * what is missing and a changed crop gets a new name instead of
 * replacing a URL a reader may have cached.
 *
 * Then: `<images>/<book>/images.urls.json` ({repo, releases, urls:
 * {sha256: url}}) -- it lives in the data repository and travels by git --
 * and `UPDATE images SET url` in the database, so `32 --urls
 * '<data>/notations/*\/images.urls.json' --require-urls` carries them into
 * the next build.
 *
 * Run it on the machine that cut the images (the one that ran 27/29 on
 * the book): a crop is byte-identical only on the same kind of machine,
 * since the deskew rotation rounds differently on Apple silicon and on
 * x86, so every file is checked against its record's sha256 and one that
 * differs is skipped and counted, never uploaded. Only the URLs travel,
 * by git, in images.urls.json.
 *
 * The repository is --repo, or NOTATION_ASSETS_REPO in the environment.
 * Needs the GitHub CLI (`gh`) signed in with write access, and the
 * repository to exist (`gh repo create <repo> --public`). --dry-run plans
 * and touches nothing; --max caps the uploads of one run (GitHub limits
 * how fast content may be created; a stopped run resumes and exits 3);
 * --verify fetches every published URL (or --sample N of them) and fails
 * on any that does not answer 200 or any shown image without a URL;
 * --prune deletes the assets of a book's releases that no image names any
 * more (otherwise they are only counted).
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';

const args = process.argv.slice(2);
const opt = (name, dflt = null) => { const i = args.indexOf(name); return i >= 0 && i + 1 < args.length ? args[i + 1] : dflt; };
const opts = name => args.flatMap((a, i) => (a === name && i + 1 < args.length ? [args[i + 1]] : []));
const flag = name => args.includes(name);
const num = (name, dflt) => {
  const v = opt(name);
  if (v === null) return dflt;
  const n = Number(v);
  if (!Number.isInteger(n) || n < 1) { console.error(`${name} wants a positive whole number`); process.exit(2); }
  return n;
};

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const DB = path.resolve(opt('--db', path.join(process.env.ARTIFACTS_DIR || path.join(ROOT, 'artifacts'), 'notations.sqlite')));
const IMAGES = path.resolve(opt('--images', process.env.NOTATIONS_DIR || path.join(ROOT, 'data', 'notations')));
const REPO = opt('--repo', process.env.NOTATION_ASSETS_REPO || null);
const ONLY = new Set(opts('--book'));
const ALL_ROLES = flag('--all-roles');
const DRY = flag('--dry-run');
const VERIFY = flag('--verify');
const PRUNE = flag('--prune');
const MAX = num('--max', Infinity);
const PER_RELEASE = num('--per-release', 900);
const BATCH = num('--batch', 25);
const SAMPLE = num('--sample', 0);
const RETRIES = 5;
const RETRY_BASE_S = Number(process.env.PUBLISH_RETRY_BASE_S || 30);   // the test sets it to 0
const sleep = ms => new Promise(r => setTimeout(r, ms));

function gh(argv, { json = false } = {}) {
  const out = execFileSync('gh', argv, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 64 << 20 });
  return json ? JSON.parse(out) : out;
}

/** The images of one book the app shows: a notation's block cuts and thumbs, or all of a notation without blocks. */
function shown(rows) {
  if (ALL_ROLES) return rows;
  const byNotation = new Map();
  for (const r of rows) {
    if (!byNotation.has(r.notation_id)) byNotation.set(r.notation_id, []);
    byNotation.get(r.notation_id).push(r);
  }
  const out = [];
  for (const group of byNotation.values()) {
    const blocks = group.filter(r => r.role === 'block');
    out.push(...(blocks.length ? blocks : group));
  }
  return out;
}

const sha256Of = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const assetName = row => `${row.notation_id.replace(/:/g, '-')}-${row.n}-${row.kind}-${row.sha256.slice(0, 8)}.png`;
const tagOf = (book, k) => (k === 1 ? `notations-${book}-v1` : `notations-${book}-v1-p${k}`);

/** Every release of the repository, by tag. */
function releaseTags() {
  return new Set(gh(['release', 'list', '--repo', REPO, '--limit', '1000', '--json', 'tagName'], { json: true }).map(r => r.tagName));
}

/** One release's assets, every page of them: [{name, url, state}]. */
function assetsOf(tag) {
  const id = gh(['api', `repos/${REPO}/releases/tags/${tag}`, '--jq', '.id']).trim();
  const out = gh(['api', '--paginate', `repos/${REPO}/releases/${id}/assets?per_page=100`,
                  '--jq', '.[] | [.name, .browser_download_url, .state] | @tsv']);
  return out.split('\n').filter(Boolean).map(l => { const [name, url, state] = l.split('\t'); return { name, url, state }; });
}

async function withRetries(what, fn) {
  for (let attempt = 1; ; attempt++) {
    try { return fn(); } catch (err) {
      const msg = String(err.stderr || err.message || err).trim().split('\n').slice(-2).join(' ');
      if (attempt >= RETRIES) throw new Error(`${what}: ${msg}`);
      const wait = RETRY_BASE_S * 2 ** (attempt - 1);
      console.log(`  ${what} failed (${msg.slice(0, 160)}); retrying in ${wait}s`);
      await sleep(wait * 1000);
    }
  }
}

async function verify(rows) {
  const urls = [...new Set(rows.map(r => r.url).filter(Boolean))];
  const missing = rows.filter(r => !r.url).length;
  const pick = SAMPLE && SAMPLE < urls.length ? urls.sort(() => Math.random() - 0.5).slice(0, SAMPLE) : urls;
  let bad = 0;
  for (let i = 0; i < pick.length; i += 8) {
    const got = await Promise.all(pick.slice(i, i + 8).map(async u => {
      try { const r = await fetch(u, { method: 'HEAD', redirect: 'follow' }); return [u, r.status]; } catch { return [u, 0]; }
    }));
    for (const [u, status] of got) if (status !== 200) { bad += 1; console.log(`  ${status || 'no answer'}: ${u}`); }
  }
  console.log(`${pick.length} url(s) fetched, ${bad} failing; ${missing} shown image(s) without a url`);
  return bad === 0 && missing === 0;
}

async function main() {
  if (!fs.existsSync(DB)) { console.error(`no database at ${DB}; run 32_build_notations_db.py first`); process.exit(1); }
  const db = new DatabaseSync(DB, { readOnly: VERIFY || DRY });
  const all = db.prepare('SELECT notation_id, n, kind, role, path, url, sha256, bytes FROM images ORDER BY notation_id, n, kind').all()
    .filter(r => !ONLY.size || ONLY.has(r.notation_id.split(':')[0]));
  const rows = shown(all);
  const books = [...new Set(rows.map(r => r.notation_id.split(':')[0]))];
  console.log(`${rows.length} of ${all.length} image(s) shown by the app, in ${books.length} book(s)${REPO ? ` -> ${REPO}` : ''}`);

  if (VERIFY) { db.close(); process.exit((await verify(rows)) ? 0 : 1); }
  if (!REPO) { console.error('name the assets repository: --repo OWNER/NAME, or NOTATION_ASSETS_REPO in the environment'); process.exit(2); }

  let exists = true;
  try { gh(['repo', 'view', REPO, '--json', 'name']); } catch { exists = false; }
  const makeIt = `gh repo create ${REPO} --public --description "Scan crops of keertan notation books, served as release assets"`;
  if (!exists && !DRY) { console.error(`no repository ${REPO}: create it first with\n  ${makeIt}`); process.exit(1); }
  if (!exists) console.log(`(no repository ${REPO} yet: planning against an empty one; it is made with  ${makeIt})`);
  const tags = exists ? releaseTags() : new Set();
  const update = DRY ? null : db.prepare('UPDATE images SET url = ? WHERE sha256 = ? AND notation_id LIKE ?');
  let uploaded = 0, kept = 0, orphans = 0, elsewhere = 0, stopped = false;
  for (const book of books) {
    const mine = rows.filter(r => r.notation_id.startsWith(book + ':'));
    // one asset a content hash: the first row that carries it names it
    const want = new Map();
    for (const r of mine) if (r.sha256 && !want.has(r.sha256)) want.set(r.sha256, { row: r, name: assetName(r), file: path.join(IMAGES, r.path) });
    const shaOfName = new Map([...want.values()].map(w => [w.name, w.row.sha256]));
    const shards = [];
    for (let k = 1; tags.has(tagOf(book, k)); k++) {
      const tag = tagOf(book, k);
      shards.push({ tag, assets: await withRetries(`list ${tag}`, () => assetsOf(tag)) });
    }
    const have = new Set();                              // names published, across the book's releases, finished uploads only
    for (const s of shards) for (const a of s.assets) if (a.state === 'uploaded') have.add(a.name);
    const todo = [];
    const elsewhereBefore = elsewhere, keptBefore = kept;
    for (const w of want.values()) {
      if (have.has(w.name)) { kept += 1; continue; }
      if (!fs.existsSync(w.file)) { continue; }        // counted below: not cut on this machine, or not re-cut yet
      // a crop is reproducible only on the kind of machine that cut it (the deskew rotation rounds differently on
      // Apple silicon and on x86): a file whose bytes are not the record's is never uploaded under the record's name
      if (sha256Of(w.file) !== w.row.sha256) { elsewhere += 1; continue; }
      todo.push(w);
    }
    const stale = shards.flatMap(s => s.assets.filter(a => !shaOfName.has(a.name)).map(a => ({ tag: s.tag, name: a.name })));
    orphans += stale.length;
    const gone = want.size - todo.length - (kept - keptBefore) - (elsewhere - elsewhereBefore);
    console.log(`${book}: ${want.size} image(s): ${kept - keptBefore} published, ${todo.length} to upload`
                + `${elsewhere - elsewhereBefore ? `, ${elsewhere - elsewhereBefore} cut on another machine` : ''}`
                + `${gone ? `, ${gone} not on this machine (cut elsewhere, or 29_notation_parse.py --recut --book ${book})` : ''}; `
                + `${shards.length} release(s)${stale.length ? `, ${stale.length} asset(s) no image names` : ''}`);

    let k = shards.length;
    let room = k ? PER_RELEASE - shards[k - 1].assets.length : 0;
    while (todo.length && uploaded < MAX) {
      if (room <= 0) {
        k += 1;
        const tag = tagOf(book, k);
        console.log(`  ${tag}: creating the release`);
        if (!DRY) {
          await withRetries(`create ${tag}`, () => gh(['release', 'create', tag, '--repo', REPO,
            '--title', `Notation images: ${book}${k > 1 ? ` (part ${k})` : ''}`,
            '--notes', `Scan crops of ${book}, a keertan notation book, as the notations database of the Gurbani app points at them. `
                       + 'Built by 32_build_notations_db.py, uploaded by tools/publish-notation-images.mjs. One asset per image, named after its content.']));
        }
        tags.add(tag);
        shards.push({ tag, assets: [] });
        room = PER_RELEASE;
      }
      const tag = tagOf(book, k);
      const batch = todo.splice(0, Math.min(BATCH, room, MAX - uploaded));
      if (DRY) {
        console.log(`  would upload ${batch.length} to ${tag}: ${batch.slice(0, 2).map(b => b.name).join(', ')}${batch.length > 2 ? ', ...' : ''}`);
      } else {
        // gh names an asset after its file: each is staged under its asset name
        const stage = fs.mkdtempSync(path.join(os.tmpdir(), 'notation-upload-'));
        const staged = batch.map(b => { const p = path.join(stage, b.name); fs.copyFileSync(b.file, p); return p; });
        try {
          await withRetries(`upload ${batch.length} to ${tag}`, () => gh(['release', 'upload', tag, '--repo', REPO, '--clobber', ...staged]));
        } finally {
          fs.rmSync(stage, { recursive: true, force: true });
        }
      }
      shards[k - 1].assets.push(...batch.map(b => ({ name: b.name, state: 'uploaded' })));
      room -= batch.length;
      uploaded += batch.length;
    }
    if (todo.length) { stopped = true; console.log(`  ${todo.length} left for the next run (--max ${MAX})`); }
    if (PRUNE && stale.length && !DRY) {
      for (const s of stale) await withRetries(`delete ${s.name}`, () => gh(['release', 'delete-asset', s.tag, s.name, '--repo', REPO, '--yes']));
      console.log(`  ${stale.length} asset(s) no image names deleted`);
    }
    if (DRY) continue;

    // the URLs as GitHub now has them, into the data and the database
    const urls = {};
    for (const s of shards) {
      for (const a of await withRetries(`list ${s.tag}`, () => assetsOf(s.tag))) {
        if (a.state === 'uploaded' && shaOfName.has(a.name)) urls[shaOfName.get(a.name)] = a.url;
      }
    }
    for (const [sha, url] of Object.entries(urls)) update.run(url, sha, book + ':%');
    const out = path.join(IMAGES, book, 'images.urls.json');
    let prior = {};
    if (fs.existsSync(out)) { try { prior = JSON.parse(fs.readFileSync(out, 'utf8')).urls || {}; } catch { prior = {}; } }
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, JSON.stringify({ repo: REPO, releases: shards.map(s => s.tag), urls: { ...prior, ...urls } }, null, 1) + '\n');
    console.log(`  ${Object.keys(urls).length} url(s) -> ${out} and the database`);
  }
  if (!DRY) {
    db.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES ('images_release_base', ?)").run(`https://github.com/${REPO}/releases/download/`);
    const n = db.prepare('SELECT COUNT(*) c FROM images WHERE url IS NOT NULL').get().c;
    db.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES ('images_published', ?)").run(String(n));
  }
  db.close();
  if (elsewhere) {
    console.log(`${elsewhere} image(s) here are not the bytes their records name: they were cut on another machine; `
                + 'run this tool there (where 27/29 made them) to publish them');
  }
  console.log(`${uploaded} ${DRY ? 'to upload' : 'uploaded'}, ${kept} already there`
              + `${orphans && !PRUNE ? `, ${orphans} asset(s) no image names (--prune deletes them)` : ''}${DRY ? ' (dry run)' : ''}`);
  if (stopped) process.exitCode = 3;                  // more to do: run it again
}

main().catch(err => { console.error(err.message || err); process.exit(1); });
