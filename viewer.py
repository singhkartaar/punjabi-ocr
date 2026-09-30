#!/usr/bin/env python3
"""
Interactive Web Viewer for Santhya Sri Guru Granth Sahib Ji (Volume 1) OCR Review.
Serves scan images, bounding boxes, column layout, multi-engine comparisons,
ingested passages, and reviewer verification notes.
"""

import os
import sys
import glob
import json
import sqlite3
import urllib.parse
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_DIR = os.path.join(BASE_DIR, "data", "ocr", "santhya-vol-1")
PAGES_DIR = os.path.join(OCR_DIR, "pages")
MERGED_DIR = os.path.join(OCR_DIR, "merged")
WRITINGS_FILE = os.path.join(BASE_DIR, "data", "writings", "santhya.jsonl")
DB_FILE = os.path.join(BASE_DIR, "artifacts", "writings-pa.sqlite")
REVIEW_FILE = os.path.join(OCR_DIR, "review_status.json")
EVAL_FILE = os.path.join(BASE_DIR, "data", "raw", "ocr-eval-santhya-vol-1.json")
REPORT_FILE = os.path.join(BASE_DIR, "data", "raw", "ocr-report-santhya-vol-1.json")

# In-memory caches
PARAGRAPHS_CACHE = {}
REVIEW_STATUS = {}
SEARCH_INDEX = []  # list of (page, line_n, col, text)


def init_data():
    global REVIEW_STATUS, PARAGRAPHS_CACHE, SEARCH_INDEX
    # Load existing review status
    if os.path.exists(REVIEW_FILE):
        try:
            with open(REVIEW_FILE, "r", encoding="utf-8") as f:
                REVIEW_STATUS = json.load(f)
        except Exception as e:
            print(f"[viewer] Error loading review status: {e}")
            REVIEW_STATUS = {}

    # Index writings paragraphs by page
    if os.path.exists(WRITINGS_FILE):
        print("[viewer] Indexing writings paragraphs...")
        try:
            with open(WRITINGS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "_meta" in row:
                        continue
                    pg = row.get("page")
                    if pg:
                        if pg not in PARAGRAPHS_CACHE:
                            PARAGRAPHS_CACHE[pg] = []
                        PARAGRAPHS_CACHE[pg].append(row)
        except Exception as e:
            print(f"[viewer] Error loading writings: {e}")

    # Build search index from merged JSONL files
    print("[viewer] Indexing OCR lines for search...")
    merged_files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.jsonl")))
    for mpath in merged_files:
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                first = f.readline()
                meta = json.loads(first).get("_meta", {})
                pg = meta.get("page", 0)
                for line in f:
                    if not line.strip():
                        continue
                    ld = json.loads(line)
                    txt = ld.get("text", "")
                    if txt:
                        SEARCH_INDEX.append((pg, ld.get("n", 0), ld.get("col"), txt))
        except Exception:
            pass
    print(f"[viewer] Ready: {len(merged_files)} pages, {len(SEARCH_INDEX)} lines indexed.")


def save_review_status():
    try:
        with open(REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(REVIEW_STATUS, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[viewer] Error saving review status: {e}")


def get_sqlite_units(page_num):
    if not os.path.exists(DB_FILE):
        return []
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "SELECT unit_id, unit_row, para_no, marker, text FROM units WHERE page = ? ORDER BY para_no ASC",
            (page_num,),
        )
        rows = c.fetchall()
        conn.close()
        return [
            {"unit_id": r[0], "unit_row": r[1], "para_no": r[2], "marker": r[3], "text": r[4]}
            for r in rows
        ]
    except Exception as e:
        print(f"[viewer] SQLite query error: {e}")
        return []


def get_page_data(page_num):
    fname = f"{page_num:04d}.jsonl"
    mpath = os.path.join(MERGED_DIR, fname)
    if not os.path.exists(mpath):
        return None

    meta = {}
    lines = []
    with open(mpath, "r", encoding="utf-8") as f:
        meta_line = f.readline()
        if meta_line:
            meta = json.loads(meta_line).get("_meta", {})
        for line in f:
            if line.strip():
                lines.append(json.loads(line))

    paragraphs = PARAGRAPHS_CACHE.get(page_num, [])
    sqlite_units = get_sqlite_units(page_num)
    review = REVIEW_STATUS.get(str(page_num), {"status": "unreviewed", "notes": ""})

    has_image = os.path.exists(os.path.join(PAGES_DIR, f"{page_num:04d}.png"))

    # Compute paragraph-level alignment
    sys.path.insert(0, os.path.join(BASE_DIR, "pipeline", "python"))
    from lib.ocr_pairs import align_page_pairs
    aligned = align_page_pairs(lines, page_h=meta.get("page_h", 3000), page_w=meta.get("page_w", 2000))

    return {
        "page": page_num,
        "meta": meta,
        "lines": lines,
        "paragraphs": paragraphs,
        "sqlite_units": sqlite_units,
        "review": review,
        "has_image": has_image,
        "aligned": aligned,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="pa" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Santhya Sri Guru Granth Sahib Ji — Vol 1 OCR Review</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#f0fdf4',
              100: '#dcfce7',
              500: '#22c55e',
              600: '#16a34a',
              700: '#15803d',
            }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta+Mahee:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .gurmukhi {
      font-family: 'Mukta Mahee', 'Gurmukhi MN', 'Noto Sans Gurmukhi', sans-serif;
    }
    /* Custom scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    .bbox-rect {
      cursor: pointer;
      transition: all 0.15s ease-in-out;
    }
    .bbox-rect:hover, .bbox-rect.highlighted {
      stroke: #f59e0b !important;
      stroke-width: 3px !important;
      fill: rgba(245, 158, 11, 0.28) !important;
    }
    .line-card.highlighted {
      border-color: #f59e0b !important;
      background-color: rgba(245, 158, 11, 0.12) !important;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 flex flex-col h-screen overflow-hidden">

  <!-- TOP HEADER -->
  <header class="bg-slate-900 border-b border-slate-800 px-4 py-2.5 flex items-center justify-between flex-shrink-0 z-30 shadow-md">
    <div class="flex items-center space-x-3">
      <div class="bg-emerald-500/20 text-emerald-400 p-2 rounded-xl border border-emerald-500/30">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path></svg>
      </div>
      <div>
        <div class="flex items-center space-x-2">
          <h1 class="text-sm font-bold tracking-tight text-white">Santhya Sri Guru Granth Sahib Ji — Vol. 1</h1>
          <span class="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">Bhai Vir Singh</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800/60 font-mono">96.0% Acc (3.3% CER)</span>
        </div>
        <div class="flex items-center space-x-3 text-xs text-slate-400 mt-0.5">
          <span>530 Pages</span>
          <span>•</span>
          <span>2,619 Vector Units</span>
          <span>•</span>
          <span>12,867 Paragraphs</span>
        </div>
      </div>
    </div>

    <!-- Page navigation controls in header -->
    <div class="flex items-center space-x-2">
      <button onclick="changePage(1)" title="First Page (Home)" class="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7"/></svg>
      </button>
      <button onclick="changePage(currentPage - 1)" title="Previous Page (Left Arrow)" class="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
      </button>

      <div class="flex items-center space-x-1.5 bg-slate-800/90 px-2.5 py-1 rounded border border-slate-700">
        <span class="text-xs text-slate-400 font-medium">Page</span>
        <input id="pageInput" type="number" min="1" max="530" value="201" 
          onkeydown="if(event.key==='Enter') changePage(parseInt(this.value))"
          onchange="changePage(parseInt(this.value))"
          class="w-14 bg-slate-900 text-white text-center font-bold text-xs rounded border border-slate-700 py-0.5 focus:outline-none focus:border-emerald-500">
        <span class="text-xs text-slate-400">/ 530</span>
      </div>

      <button onclick="changePage(currentPage + 1)" title="Next Page (Right Arrow)" class="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
      </button>
      <button onclick="changePage(530)" title="Last Page (End)" class="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
      </button>

      <!-- Quick Jump Menu -->
      <select id="quickJump" onchange="changePage(parseInt(this.value))" class="bg-slate-800 text-xs text-slate-300 rounded border border-slate-700 px-2.5 py-1 focus:outline-none focus:border-emerald-500">
        <option value="">Jump to Key Sample Page...</option>
        <option value="2">Page 2 — Title & Publishing Details</option>
        <option value="3">Page 3 — Bhai Vir Singh Preface</option>
        <option value="4">Page 4 — Preface Philosophy</option>
        <option value="5">Page 5 — Preface Fire/Baesantar</option>
        <option value="45">Page 45 — Japji Sahib Early (Footnotes)</option>
        <option value="80">Page 80 — Japji Pauri 16 (Gutter Start)</option>
        <option value="201" selected>Page 201 — Sodar Rehras (2 Columns Gurbani+Trans)</option>
        <option value="280">Page 280 — Asa Di Var (Hybrid 2-Col + Prose)</option>
        <option value="350">Page 350 — Deep 2-Col (Dialogue/Frog)</option>
        <option value="420">Page 420 — Scriptural Cross-Citation</option>
        <option value="500">Page 500 — Deep 2-Column Section</option>
        <option value="530">Page 530 — Volume 1 Conclusion</option>
      </select>
    </div>

    <!-- Review Progress & Search trigger -->
    <div class="flex items-center space-x-3">
      <!-- Search input -->
      <div class="relative">
        <input id="searchInput" type="text" placeholder="Search Gurmukhi text..." 
          onkeydown="if(event.key==='Enter') doSearch()"
          class="gurmukhi w-48 bg-slate-800 text-xs text-white rounded-lg pl-8 pr-3 py-1 border border-slate-700 focus:outline-none focus:border-emerald-500 focus:w-64 transition-all">
        <svg class="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
      </div>

      <!-- Review status badge -->
      <div id="pageReviewBadge" class="flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 border border-slate-700 text-slate-400">
        <span class="w-2 h-2 rounded-full bg-slate-500"></span>
        <span>Unreviewed</span>
      </div>

      <!-- Quick summary trigger button -->
      <button onclick="toggleSummaryModal()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded border border-slate-700 flex items-center space-x-1.5 shadow-sm">
        <svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>
        <span>Review Dashboard</span>
        <span id="reviewedCountBadge" class="bg-emerald-500/20 text-emerald-300 px-1.5 rounded-full text-[10px] font-bold">0</span>
      </button>
    </div>
  </header>

  <!-- SUBHEADER / TOOLBAR -->
  <div class="bg-slate-900/90 border-b border-slate-800 px-4 py-1.5 flex items-center justify-between text-xs text-slate-400 flex-shrink-0">
    <!-- Image controls -->
    <div class="flex items-center space-x-4">
      <div class="flex items-center space-x-1">
        <span class="font-medium text-slate-300">Zoom:</span>
        <button onclick="zoomIn()" title="Zoom In (+)" class="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-300 font-bold border border-slate-700">+</button>
        <button onclick="zoomOut()" title="Zoom Out (-)" class="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-300 font-bold border border-slate-700">-</button>
        <button onclick="resetZoom()" title="Fit to Container" class="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-300 border border-slate-700">Fit Width</button>
        <span id="zoomLevel" class="text-slate-400 font-mono text-[11px] ml-1">100%</span>
      </div>

      <div class="h-4 w-px bg-slate-800"></div>

      <label class="flex items-center space-x-1.5 cursor-pointer">
        <input id="toggleBBoxes" type="checkbox" checked onchange="renderBBoxes()" class="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0">
        <span>Bounding Boxes <kbd class="px-1 bg-slate-800 text-[10px] border border-slate-700 rounded">b</kbd></span>
      </label>

      <label class="flex items-center space-x-1.5 cursor-pointer">
        <input id="toggleGutter" type="checkbox" checked onchange="renderBBoxes()" class="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0">
        <span>Gutter Guideline <kbd class="px-1 bg-slate-800 text-[10px] border border-slate-700 rounded">g</kbd></span>
      </label>

      <div class="h-4 w-px bg-slate-800"></div>

      <!-- Legend -->
      <div class="flex items-center space-x-3 text-[11px]">
        <div class="flex items-center space-x-1">
          <span class="w-2.5 h-2.5 rounded bg-blue-500/80 border border-blue-400"></span>
          <span class="text-blue-300 font-medium">Col 0: Gurbani Verse</span>
        </div>
        <div class="flex items-center space-x-1">
          <span class="w-2.5 h-2.5 rounded bg-emerald-500/80 border border-emerald-400"></span>
          <span class="text-emerald-300 font-medium">Col 1: Punjabi Translation</span>
        </div>
        <div class="flex items-center space-x-1">
          <span class="w-2.5 h-2.5 rounded bg-amber-500/80 border border-amber-400"></span>
          <span class="text-amber-300 font-medium">Single / Full Width</span>
        </div>
      </div>
    </div>

    <!-- Page hints & metrics -->
    <div id="pageMetaInfo" class="flex items-center space-x-3 text-slate-400 font-mono text-[11px]">
      <span id="pageDim">Dimensions: ...</span>
      <span id="colInfo">Columns: ...</span>
      <span id="boldThresh">Bold split: ...</span>
    </div>
  </div>

  <!-- MAIN SPLIT AREA -->
  <div class="flex-1 flex overflow-hidden">
    
    <!-- LEFT: SCAN IMAGE & OVERLAY VIEWPORT -->
    <div id="imageContainer" class="w-1/2 h-full bg-slate-950 overflow-auto relative flex justify-center items-start p-4 border-r border-slate-800 select-none cursor-grab active:cursor-grabbing">
      <div id="imageWrapper" class="relative origin-top transition-transform duration-75 shadow-2xl rounded">
        <img id="pageImage" class="max-w-none block rounded shadow-lg" alt="Scan Page" />
        <svg id="bboxSvg" class="absolute top-0 left-0 w-full h-full pointer-events-auto"></svg>
      </div>
      <div id="imageLoading" class="absolute inset-0 bg-slate-950/80 flex items-center justify-center">
        <div class="text-center">
          <div class="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-emerald-500 mb-2"></div>
          <p class="text-xs text-slate-400">Loading High-Res Scan...</p>
        </div>
      </div>
    </div>

    <!-- RIGHT: OCR INSPECTION & DATA PANE -->
    <div class="w-1/2 h-full flex flex-col bg-slate-900">
      
      <!-- TABS -->
      <div class="flex items-center border-b border-slate-800 px-4 pt-2 bg-slate-900/90 space-x-2 text-xs flex-shrink-0">
        <button id="tabPairsBtn" onclick="switchTab('pairs')" class="px-3 py-2 font-medium border-b-2 border-emerald-500 text-emerald-400 flex items-center space-x-1.5 transition">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>
          <span>Verse ↔ Translation Pairs</span>
          <span id="pairCountBadge" class="bg-slate-800 text-slate-400 px-1.5 py-0.2 rounded-full text-[10px]">0</span>
        </button>

        <button id="tabColumnsBtn" onclick="switchTab('columns')" class="px-3 py-2 font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-1.5 transition">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"/></svg>
          <span>Columns & OCR Lines</span>
          <span id="lineCountBadge" class="bg-slate-800 text-slate-400 px-1.5 py-0.2 rounded-full text-[10px]">0</span>
        </button>

        <button id="tabEnginesBtn" onclick="switchTab('engines')" class="px-3 py-2 font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-1.5 transition">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>
          <span>Multi-Engine Diff</span>
        </button>

        <button id="tabPassagesBtn" onclick="switchTab('passages')" class="px-3 py-2 font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-1.5 transition">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
          <span>Ingested Paragraphs</span>
          <span id="paraCountBadge" class="bg-slate-800 text-slate-400 px-1.5 py-0.2 rounded-full text-[10px]">0</span>
        </button>

        <button id="tabMetaBtn" onclick="switchTab('meta')" class="px-3 py-2 font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-1.5 transition">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>
          <span>Raw Diagnostics</span>
        </button>
      </div>

      <!-- TAB 0: VERSE <-> TRANSLATION PAIRS (PRIMARY) -->
      <div id="tabPairs" class="flex-1 overflow-y-auto p-4 space-y-3">
        <div class="bg-gradient-to-r from-blue-950/40 via-slate-900 to-emerald-950/40 border border-slate-800 rounded-lg p-2.5 text-xs flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <span class="p-1 rounded bg-blue-500/20 text-blue-400 font-bold">Stanza ↔ Commentary</span>
            <span class="text-slate-300">Paragraph-level alignment: Gurbani verses mapped to Punjabi translations.</span>
          </div>
          <span id="pairLayoutType" class="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-800 text-emerald-400 border border-slate-700">Two-Column Page</span>
        </div>

        <div id="pairsList" class="space-y-3"></div>
      </div>

      <!-- TAB 1: COLUMNS & LINES -->
      <div id="tabColumns" class="hidden flex-1 overflow-y-auto p-4 space-y-3">
        <!-- Two-column view container if two columns exist -->
        <div id="twoColumnLayout" class="hidden">
          <div class="grid grid-cols-2 gap-3 mb-2 sticky top-0 bg-slate-900/95 py-1.5 z-10 border-b border-slate-800">
            <div class="text-xs font-semibold text-blue-400 flex items-center justify-between px-2">
              <span class="flex items-center space-x-1.5">
                <span class="w-2 h-2 rounded-full bg-blue-500"></span>
                <span>COLUMN 0: GURBANI VERSE (BOLD)</span>
              </span>
              <span id="col0Count" class="text-[10px] text-slate-400 font-mono">0 lines</span>
            </div>
            <div class="text-xs font-semibold text-emerald-400 flex items-center justify-between px-2">
              <span class="flex items-center space-x-1.5">
                <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span>COLUMN 1: PUNJABI TRANSLATION</span>
              </span>
              <span id="col1Count" class="text-[10px] text-slate-400 font-mono">0 lines</span>
            </div>
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div id="col0Lines" class="space-y-2"></div>
            <div id="col1Lines" class="space-y-2"></div>
          </div>
        </div>

        <!-- Single column or headers/other lines -->
        <div id="singleColumnLayout" class="space-y-2"></div>
      </div>

      <!-- TAB 2: MULTI-ENGINE DIFF -->
      <div id="tabEngines" class="hidden flex-1 overflow-y-auto p-4 space-y-3">
        <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs mb-3 text-slate-300">
          <span class="font-bold text-emerald-400">Multi-Engine Consensus:</span>
          Tesseract Gurmukhi (Weight 0.850) and Tesseract Punjabi (Weight 0.804) ran independently on every line.
          Lines below show agreement rates and exact engine comparisons.
        </div>
        <div id="engineDiffList" class="space-y-2.5"></div>
      </div>

      <!-- TAB 3: INGESTED PARAGRAPHS -->
      <div id="tabPassages" class="hidden flex-1 overflow-y-auto p-4 space-y-3">
        <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs mb-3 flex items-center justify-between text-slate-300">
          <div>
            <span class="font-bold text-emerald-400">Ingested Output:</span>
            Final paragraphs extracted by <code class="text-slate-400">12_ingest_writings.py</code> into <code class="text-slate-400">santhya.jsonl</code> and stored in <code class="text-slate-400">writings-pa.sqlite</code>.
          </div>
        </div>
        <div id="passagesList" class="space-y-2.5"></div>
      </div>

      <!-- TAB 4: RAW METADATA -->
      <div id="tabMeta" class="hidden flex-1 overflow-y-auto p-4">
        <pre id="rawMetaJson" class="text-xs font-mono text-emerald-400 bg-slate-950 p-4 rounded-lg border border-slate-800 overflow-x-auto whitespace-pre-wrap"></pre>
      </div>

      <!-- BOTTOM REVIEW FOOTER BAR -->
      <div class="bg-slate-950 border-t border-slate-800 p-3 flex-shrink-0 flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <span class="text-xs text-slate-400 font-medium">Page Review:</span>
          <button id="btnApprove" onclick="setReviewStatus('approved')" class="px-3 py-1 bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 rounded text-xs font-semibold flex items-center space-x-1 transition">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            <span>Approve Page</span>
          </button>
          <button id="btnNeedsReview" onclick="setReviewStatus('needs_review')" class="px-3 py-1 bg-amber-950 hover:bg-amber-900 text-amber-300 border border-amber-800 rounded text-xs font-semibold flex items-center space-x-1 transition">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            <span>Needs Review</span>
          </button>
          <button id="btnFlag" onclick="setReviewStatus('flagged')" class="px-3 py-1 bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-800 rounded text-xs font-semibold flex items-center space-x-1 transition">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V9"/></svg>
            <span>Flag Issue</span>
          </button>
        </div>

        <div class="flex items-center space-x-2 flex-1 max-w-md ml-4">
          <input id="reviewNoteInput" type="text" placeholder="Add reviewer note for this page..." 
            onchange="saveReviewNote(this.value)"
            class="w-full bg-slate-900 text-xs text-white rounded border border-slate-700 px-3 py-1 focus:outline-none focus:border-emerald-500">
          <button onclick="saveReviewNote(document.getElementById('reviewNoteInput').value); changePage(currentPage + 1)" class="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded text-xs whitespace-nowrap">
            Save & Next &rarr;
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- SUMMARY & SEARCH MODAL -->
  <div id="modalOverlay" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl">
      <div class="flex items-center justify-between border-b border-slate-800 px-5 py-3">
        <h3 id="modalTitle" class="text-sm font-bold text-white">Review Summary & Notes</h3>
        <button onclick="closeModal()" class="text-slate-400 hover:text-white text-lg">&times;</button>
      </div>
      <div id="modalBody" class="flex-1 overflow-y-auto p-5 space-y-4">
        <!-- Content inserted dynamically -->
      </div>
      <div class="border-t border-slate-800 px-5 py-3 flex items-center justify-between">
        <button id="btnExportJson" onclick="exportReviewJson()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold rounded border border-slate-700 flex items-center space-x-1.5">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
          <span>Download Reviews JSON</span>
        </button>
        <button onclick="closeModal()" class="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-semibold rounded border border-slate-700">Close</button>
      </div>
    </div>
  </div>

  <script>
    let currentPage = 201;
    let pageData = null;
    let currentZoom = 1.0;
    let currentTab = 'pairs';
    let isDragging = false;
    let startX, startY, scrollLeft, scrollTop;

    // Pan interaction
    const container = document.getElementById('imageContainer');
    container.addEventListener('mousedown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.classList.contains('bbox-rect')) return;
      isDragging = true;
      startX = e.pageX - container.offsetLeft;
      startY = e.pageY - container.offsetTop;
      scrollLeft = container.scrollLeft;
      scrollTop = container.scrollTop;
    });
    container.addEventListener('mouseleave', () => isDragging = false);
    container.addEventListener('mouseup', () => isDragging = false);
    container.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      e.preventDefault();
      const x = e.pageX - container.offsetLeft;
      const y = e.pageY - container.offsetTop;
      container.scrollLeft = scrollLeft - (x - startX);
      container.scrollTop = scrollTop - (y - startY);
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
      if (e.key === 'ArrowLeft') changePage(currentPage - 1);
      if (e.key === 'ArrowRight') changePage(currentPage + 1);
      if (e.key === 'Home') changePage(1);
      if (e.key === 'End') changePage(530);
      if (e.key === 'b') {
        const cb = document.getElementById('toggleBBoxes');
        cb.checked = !cb.checked;
        renderBBoxes();
      }
      if (e.key === 'g') {
        const cb = document.getElementById('toggleGutter');
        cb.checked = !cb.checked;
        renderBBoxes();
      }
      if (e.key === '+' || e.key === '=') zoomIn();
      if (e.key === '-' || e.key === '_') zoomOut();
    });

    // Resize observer to keep SVG aligned to image
    const imgEl = document.getElementById('pageImage');
    const resizeObserver = new ResizeObserver(() => {
      renderBBoxes();
    });
    resizeObserver.observe(imgEl);

    async function changePage(p) {
      if (isNaN(p) || p < 1) p = 1;
      if (p > 530) p = 530;
      currentPage = p;
      if (window.location.hash !== `#${p}`) {
        window.history.replaceState(null, null, `#${p}`);
      }
      document.getElementById('pageInput').value = p;
      document.getElementById('imageLoading').style.display = 'flex';

      try {
        const res = await fetch(`/api/page/${p}`);
        if (!res.ok) throw new Error('Page not found');
        pageData = await res.json();
        renderPage();
      } catch (err) {
        console.error(err);
      }
    }

    function renderPage() {
      if (!pageData) return;
      const meta = pageData.meta || {};
      const lines = pageData.lines || [];
      const paragraphs = pageData.paragraphs || [];

      // Update Header & Meta
      document.getElementById('pageDim').textContent = `${meta.page_w || '?'} x ${meta.page_h || '?'} px`;
      document.getElementById('colInfo').textContent = meta.columns ? `Columns: ${meta.columns.length} (${meta.columns.map(c => (c[1]-c[0])+'px').join(' | ')})` : 'Columns: Single';
      document.getElementById('boldThresh').textContent = `Bold split: ${meta.bold_split || '?'}px (body ${meta.body_h || '?'}px)`;
      document.getElementById('lineCountBadge').textContent = lines.length;
      document.getElementById('paraCountBadge').textContent = paragraphs.length;

      // Update Review Bar
      const rev = pageData.review || { status: 'unreviewed', notes: '' };
      updateReviewUI(rev.status, rev.notes);

      // Load Image
      const img = document.getElementById('pageImage');
      img.onload = () => {
        document.getElementById('imageLoading').style.display = 'none';
        renderBBoxes();
      };
      img.src = `/api/page/${currentPage}/image`;

      // Render Tabs
      renderPairsTab(pageData.aligned);
      renderColumnsTab(lines, meta);
      renderEnginesTab(lines);
      renderPassagesTab(paragraphs, pageData.sqlite_units);
      document.getElementById('rawMetaJson').textContent = JSON.stringify(meta, null, 2);
    }

    function renderBBoxes() {
      if (!pageData) return;
      const svg = document.getElementById('bboxSvg');
      svg.innerHTML = '';
      const showBoxes = document.getElementById('toggleBBoxes').checked;
      const showGutter = document.getElementById('toggleGutter').checked;

      const img = document.getElementById('pageImage');
      const w = img.clientWidth;
      const h = img.clientHeight;
      if (!w || !h) return;

      const meta = pageData.meta || {};
      const scaleX = w / (meta.page_w || w);
      const scaleY = h / (meta.page_h || h);

      // Draw Gutter line if 2 columns exist
      if (showGutter && meta.columns && meta.columns.length > 1) {
        const gutterX = meta.columns[0][1] * scaleX;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', gutterX);
        line.setAttribute('y1', 0);
        line.setAttribute('x2', gutterX);
        line.setAttribute('y2', h);
        line.setAttribute('stroke', '#06b6d4');
        line.setAttribute('stroke-width', '2.5');
        line.setAttribute('stroke-dasharray', '8 4');
        svg.appendChild(line);

        // Gutter label
        const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        txt.setAttribute('x', gutterX + 6);
        txt.setAttribute('y', 36);
        txt.setAttribute('fill', '#06b6d4');
        txt.setAttribute('font-size', '12');
        txt.setAttribute('font-family', 'sans-serif');
        txt.setAttribute('font-weight', 'bold');
        txt.textContent = 'GUTTER MARGIN';
        svg.appendChild(txt);
      }

      if (!showBoxes) return;

      const lines = pageData.lines || [];
      lines.forEach((l) => {
        if (!l.bbox) return;
        const [x0, y0, x1, y1] = l.bbox;
        const rx = x0 * scaleX;
        const ry = y0 * scaleY;
        const rw = (x1 - x0) * scaleX;
        const rh = (y1 - y0) * scaleY;

        let stroke = '#f59e0b';
        let fill = 'rgba(245, 158, 11, 0.08)';
        if (l.col === 0) {
          stroke = '#3b82f6'; // Gurbani
          fill = 'rgba(59, 130, 246, 0.12)';
        } else if (l.col === 1) {
          stroke = '#10b981'; // Translation
          fill = 'rgba(16, 185, 129, 0.12)';
        }

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', rx);
        rect.setAttribute('y', ry);
        rect.setAttribute('width', rw);
        rect.setAttribute('height', rh);
        rect.setAttribute('stroke', stroke);
        rect.setAttribute('stroke-width', '1.5');
        rect.setAttribute('fill', fill);
        rect.setAttribute('class', 'bbox-rect');
        rect.setAttribute('id', `bbox-${l.n}`);

        rect.addEventListener('mouseenter', () => highlightLine(l.n, true));
        rect.addEventListener('mouseleave', () => highlightLine(l.n, false));
        rect.addEventListener('click', () => {
          const el = document.getElementById(`line-${l.n}`);
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });

        // SVG Tooltip
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `Line ${l.n} [Col ${l.col !== undefined ? l.col : '-'}] stroke: ${l.stroke || '?'}px\\n${l.text}`;
        rect.appendChild(title);

        svg.appendChild(rect);
      });
    }

    function highlightLine(lineNum, isHighlight) {
      const bbox = document.getElementById(`bbox-${lineNum}`);
      const lineCard = document.getElementById(`line-${lineNum}`);
      if (bbox) {
        if (isHighlight) bbox.classList.add('highlighted');
        else bbox.classList.remove('highlighted');
      }
      if (lineCard) {
        if (isHighlight) {
          lineCard.classList.add('highlighted');
          lineCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
          lineCard.classList.remove('highlighted');
        }
      }
    function renderPairsTab(aligned) {
      const container = document.getElementById('pairsList');
      container.innerHTML = '';
      if (!aligned) return;

      const pairs = aligned.pairs || [];
      document.getElementById('pairCountBadge').textContent = pairs.length;
      document.getElementById('pairLayoutType').textContent = aligned.is_two_col ? 'Two-Column Verse Layout' : 'Single-Column Prose Layout';

      // Header lines if present
      if (aligned.header && aligned.header.length > 0) {
        const headDiv = document.createElement('div');
        headDiv.className = 'bg-slate-950/80 border border-slate-800 p-2.5 rounded-lg text-center text-xs text-amber-300 font-semibold gurmukhi';
        headDiv.textContent = aligned.header.map(l => l.text).join(' | ');
        container.appendChild(headDiv);
      }

      if (pairs.length === 0) {
        container.innerHTML += '<div class="text-center text-slate-500 text-xs py-8">No text blocks found on this page.</div>';
        return;
      }

      pairs.forEach((p) => {
        const card = createPairCard(p, aligned.is_two_col);
        container.appendChild(card);
      });

      // Footnote lines if present
      if (aligned.footnotes && aligned.footnotes.length > 0) {
        const footDiv = document.createElement('div');
        footDiv.className = 'bg-slate-950/80 border border-slate-800 p-2.5 rounded-lg text-xs text-rose-300 gurmukhi space-y-1';
        footDiv.innerHTML = '<div class="text-[10px] font-mono uppercase text-slate-400 font-bold mb-1">Footnotes & Notes:</div>' +
          aligned.footnotes.map(l => `<p>${escapeHtml(l.text)}</p>`).join('');
        container.appendChild(footDiv);
      }
    }

    function createPairCard(p, isTwoCol) {
      const div = document.createElement('div');
      div.className = 'pair-card bg-slate-950/70 border border-slate-800 hover:border-slate-600 rounded-xl p-3.5 transition duration-150 shadow-sm cursor-pointer';
      div.id = `pair-${p.pair_id}`;

      const gLines = p.gurbani ? p.gurbani.lines : [];
      const tLines = p.translation ? p.translation.lines : [];
      const allLineNums = [...gLines.map(l => l.n), ...tLines.map(l => l.n)];

      div.addEventListener('mouseenter', () => highlightPair(allLineNums, true));
      div.addEventListener('mouseleave', () => highlightPair(allLineNums, false));
      div.addEventListener('click', () => {
        const img = document.getElementById('pageImage');
        const container = document.getElementById('imageContainer');
        const meta = pageData.meta || {};
        const scaleY = img.clientHeight / (meta.page_h || img.clientHeight);
        container.scrollTo({ top: p.y_range[0] * scaleY - 60, behavior: 'smooth' });
      });

      if (isTwoCol && p.gurbani) {
        div.innerHTML = `
          <div class="flex items-center justify-between pb-2 mb-2.5 border-b border-slate-800 text-[11px] text-slate-400 font-mono">
            <span class="font-bold text-amber-400 flex items-center space-x-1.5">
              <span>Section ${p.pair_id}</span>
              <span class="text-slate-500 font-normal">| y=[${p.y_range[0]}..${p.y_range[1]}]</span>
            </span>
            <div class="flex items-center space-x-3 text-[10px]">
              <span class="text-blue-400 font-semibold">Gurbani: ${gLines.length} line(s)</span>
              <span class="text-emerald-400 font-semibold">Translation: ${tLines.length} line(s)</span>
            </div>
          </div>
          <div class="grid grid-cols-12 gap-3 items-stretch">
            <!-- Left: Gurbani Stanza -->
            <div class="col-span-5 bg-blue-950/20 border border-blue-900/60 rounded-lg p-3 flex flex-col justify-between">
              <div>
                <div class="text-[10px] font-bold text-blue-400 uppercase tracking-wider mb-1.5 flex items-center justify-between">
                  <span class="flex items-center space-x-1">
                    <span class="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                    <span>Gurbani Verse</span>
                  </span>
                  <span class="font-mono text-slate-400 text-[9px]">${gLines.map(l=>'#'+l.n).join(', ')}</span>
                </div>
                <div class="gurmukhi text-base leading-relaxed font-bold text-blue-100 space-y-1">
                  ${gLines.map(l => `<p class="hover:text-amber-300 transition">${escapeHtml(l.text)}</p>`).join('')}
                </div>
              </div>
              <div class="mt-2 text-[10px] text-slate-500 font-mono text-right">
                ${gLines.length > 0 && gLines[0].conf ? 'conf: ' + Math.round(gLines[0].conf*100)+'%' : ''}
              </div>
            </div>

            <!-- Divider arrow -->
            <div class="col-span-1 flex items-center justify-center text-slate-600">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
            </div>

            <!-- Right: Punjabi Translation -->
            <div class="col-span-6 bg-emerald-950/20 border border-emerald-900/60 rounded-lg p-3 flex flex-col justify-between">
              <div>
                <div class="text-[10px] font-bold text-emerald-400 uppercase tracking-wider mb-1.5 flex items-center justify-between">
                  <span class="flex items-center space-x-1">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                    <span>Punjabi Translation / Viakhya</span>
                  </span>
                  <span class="font-mono text-slate-400 text-[9px]">${tLines.map(l=>'#'+l.n).join(', ')}</span>
                </div>
                <p class="gurmukhi text-sm leading-relaxed text-slate-200">
                  ${escapeHtml(p.translation.text)}
                </p>
              </div>
              <div class="mt-2 text-[10px] text-slate-500 font-mono text-right">
                ${tLines.length > 0 && tLines[0].conf ? 'conf: ' + Math.round(tLines[0].conf*100)+'%' : ''}
              </div>
            </div>
          </div>
        `;
      } else {
        div.innerHTML = `
          <div class="flex items-center justify-between pb-2 mb-2 border-b border-slate-800 text-[11px] text-slate-400 font-mono">
            <span class="font-bold text-emerald-400">Paragraph ${p.pair_id}</span>
            <span class="text-slate-500">y=[${p.y_range[0]}..${p.y_range[1]}] • ${tLines.length} line(s)</span>
          </div>
          <p class="gurmukhi text-sm leading-relaxed text-slate-200">
            ${escapeHtml(p.translation.text)}
          </p>
        `;
      }

      return div;
    }

    function highlightPair(lineNums, isHighlight) {
      lineNums.forEach(n => highlightLine(n, isHighlight));
    }

    function renderColumnsTab(lines, meta) {
      const twoColLayout = document.getElementById('twoColumnLayout');
      const singleColLayout = document.getElementById('singleColumnLayout');
      const col0Container = document.getElementById('col0Lines');
      const col1Container = document.getElementById('col1Lines');

      col0Container.innerHTML = '';
      col1Container.innerHTML = '';
      singleColLayout.innerHTML = '';

      const isTwoColumn = meta.columns && meta.columns.length > 1;

      let col0Count = 0;
      let col1Count = 0;

      lines.forEach((l) => {
        const card = createLineCard(l);
        if (isTwoColumn && (l.col === 0 || l.col === 1)) {
          if (l.col === 0) {
            col0Container.appendChild(card);
            col0Count++;
          } else {
            col1Container.appendChild(card);
            col1Count++;
          }
        } else {
          singleColLayout.appendChild(card);
        }
      });

      if (isTwoColumn && (col0Count > 0 || col1Count > 0)) {
        twoColLayout.classList.remove('hidden');
        document.getElementById('col0Count').textContent = `${col0Count} lines`;
        document.getElementById('col1Count').textContent = `${col1Count} lines`;
      } else {
        twoColLayout.classList.add('hidden');
      }
    }

    function createLineCard(l) {
      const div = document.createElement('div');
      div.className = `line-card p-2.5 rounded-lg border text-xs transition duration-150 cursor-pointer ${
        l.col === 0 
          ? 'bg-blue-950/20 border-blue-900/60 hover:border-blue-500' 
          : l.col === 1 
          ? 'bg-emerald-950/20 border-emerald-900/60 hover:border-emerald-500' 
          : 'bg-slate-950/40 border-slate-800 hover:border-slate-600'
      }`;
      div.id = `line-${l.n}`;

      div.addEventListener('mouseenter', () => highlightLine(l.n, true));
      div.addEventListener('mouseleave', () => highlightLine(l.n, false));

      const isBold = l.bold;
      const stroke = l.stroke ? l.stroke.toFixed(2) + 'px' : '-';
      const conf = l.conf ? Math.round(l.conf * 100) + '%' : '-';
      const agree = l.agreement !== undefined ? Math.round(l.agreement * 100) + '%' : '100%';

      let kindBadge = '';
      if (l.kind === 'heading') kindBadge = '<span class="px-1.5 py-0.2 rounded bg-purple-900/60 text-purple-300 font-mono text-[10px]">verse</span>';
      else if (l.kind === 'commentary') kindBadge = '<span class="px-1.5 py-0.2 rounded bg-teal-900/60 text-teal-300 font-mono text-[10px]">commentary</span>';
      else if (l.kind === 'gurbani-unmatched') kindBadge = '<span class="px-1.5 py-0.2 rounded bg-blue-900/60 text-blue-300 font-mono text-[10px]">gurbani</span>';
      else if (l.zone === 'header') kindBadge = '<span class="px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 font-mono text-[10px]">header</span>';
      else if (l.zone === 'footnote') kindBadge = '<span class="px-1.5 py-0.2 rounded bg-rose-900/60 text-rose-300 font-mono text-[10px]">footnote</span>';

      div.innerHTML = `
        <div class="flex items-center justify-between mb-1 text-[11px] text-slate-400">
          <div class="flex items-center space-x-1.5">
            <span class="font-mono text-slate-500 font-semibold">#${l.n}</span>
            ${kindBadge}
            ${isBold ? '<span class="px-1 rounded bg-amber-900/50 text-amber-300 text-[10px] font-bold">BOLD</span>' : ''}
          </div>
          <div class="flex items-center space-x-2 font-mono text-[10px]">
            <span title="Stroke width">✍️ ${stroke}</span>
            <span title="Confidence" class="${l.conf < 0.85 ? 'text-amber-400' : 'text-emerald-400'}">${conf}</span>
            <span title="Agreement between engines" class="text-slate-400">agr: ${agree}</span>
          </div>
        </div>
        <p class="gurmukhi text-base leading-relaxed tracking-wide text-slate-100 ${isBold ? 'font-bold text-blue-200' : 'font-normal'}">
          ${escapeHtml(l.text || '')}
        </p>
      `;
      return div;
    }

    function renderEnginesTab(lines) {
      const container = document.getElementById('engineDiffList');
      container.innerHTML = '';

      lines.forEach((l) => {
        if (!l.raw) return;
        const engines = Object.keys(l.raw);
        const p1 = l.raw['tesseract-pan'] || '';
        const p2 = l.raw['tesseract-gurmukhi'] || '';
        const matches = (p1 === p2);

        const card = document.createElement('div');
        card.className = `p-3 rounded-lg border text-xs ${matches ? 'bg-slate-950/30 border-slate-800/80' : 'bg-amber-950/20 border-amber-800/60'}`;
        card.innerHTML = `
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center space-x-2">
              <span class="font-mono text-slate-400 font-bold">Line ${l.n}</span>
              <span class="px-1.5 py-0.2 rounded text-[10px] ${matches ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-amber-950 text-amber-300 border border-amber-800'}">
                ${matches ? '100% Match' : 'Engines Disagreed & Resolved'}
              </span>
            </div>
            <span class="text-slate-400 font-mono text-[11px]">Selected: <strong class="text-emerald-300">${escapeHtml(l.text)}</strong></span>
          </div>
          <div class="grid grid-cols-2 gap-3 text-xs">
            <div class="bg-slate-900/80 p-2 rounded border border-slate-800">
              <div class="text-[10px] text-slate-500 font-mono mb-1">tesseract-pan (Weight 0.804)</div>
              <div class="gurmukhi text-sm text-slate-200">${escapeHtml(p1)}</div>
            </div>
            <div class="bg-slate-900/80 p-2 rounded border border-slate-800">
              <div class="text-[10px] text-slate-500 font-mono mb-1">tesseract-gurmukhi (Weight 0.850)</div>
              <div class="gurmukhi text-sm text-slate-200">${escapeHtml(p2)}</div>
            </div>
          </div>
        `;
        container.appendChild(card);
      });
    }

    function renderPassagesTab(paragraphs, sqlite_units) {
      const container = document.getElementById('passagesList');
      container.innerHTML = '';

      if (!paragraphs || paragraphs.length === 0) {
        container.innerHTML = `<div class="text-xs text-slate-500 italic p-4 text-center">No paragraphs ingested for this page.</div>`;
        return;
      }

      paragraphs.forEach((p) => {
        const card = document.createElement('div');
        card.className = `p-3 rounded-lg border bg-slate-950/50 border-slate-800 text-xs hover:border-slate-700 transition`;
        card.innerHTML = `
          <div class="flex items-center justify-between mb-1.5">
            <div class="flex items-center space-x-2 font-mono text-[11px]">
              <span class="text-emerald-400 font-bold">${p.unit_id}</span>
              <span class="px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 text-[10px]">style: ${p.style || 'body'}</span>
              ${p.marker ? `<span class="px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 text-[10px]">book_page: ${p.marker}</span>` : ''}
            </div>
          </div>
          <p class="gurmukhi text-base leading-relaxed text-slate-100 ${p.style === 'quote' ? 'font-semibold text-blue-300' : 'text-slate-300'}">
            ${escapeHtml(p.text || '')}
          </p>
        `;
        container.appendChild(card);
      });
    }

    function switchTab(tab) {
      currentTab = tab;
      const tabs = ['pairs', 'columns', 'engines', 'passages', 'meta'];
      tabs.forEach(t => {
        const btn = document.getElementById(`tab${capitalize(t)}Btn`);
        const pane = document.getElementById(`tab${capitalize(t)}`);
        if (!btn || !pane) return;
        if (t === tab) {
          btn.className = 'px-3 py-2 font-medium border-b-2 border-emerald-500 text-emerald-400 flex items-center space-x-1.5 transition';
          pane.classList.remove('hidden');
        } else {
          btn.className = 'px-3 py-2 font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-1.5 transition';
          pane.classList.add('hidden');
        }
      });
    }

    function capitalize(s) {
      return s.charAt(0).toUpperCase() + s.slice(1);
    }

    function zoomIn() {
      currentZoom = Math.min(currentZoom + 0.2, 3.0);
      applyZoom();
    }

    function zoomOut() {
      currentZoom = Math.max(currentZoom - 0.2, 0.4);
      applyZoom();
    }

    function resetZoom() {
      currentZoom = 1.0;
      applyZoom();
    }

    function applyZoom() {
      const wrapper = document.getElementById('imageWrapper');
      wrapper.style.transform = `scale(${currentZoom})`;
      document.getElementById('zoomLevel').textContent = `${Math.round(currentZoom * 100)}%`;
      setTimeout(renderBBoxes, 80);
    }

    async function setReviewStatus(status) {
      const note = document.getElementById('reviewNoteInput').value;
      try {
        const res = await fetch(`/api/page/${currentPage}/review`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status, notes: note })
        });
        if (res.ok) {
          updateReviewUI(status, note);
          updateSummaryBadge();
        }
      } catch (err) {
        console.error(err);
      }
    }

    async function saveReviewNote(note) {
      const badge = document.getElementById('pageReviewBadge');
      const curStatus = pageData && pageData.review ? pageData.review.status : 'unreviewed';
      try {
        await fetch(`/api/page/${currentPage}/review`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: curStatus, notes: note })
        });
        updateSummaryBadge();
      } catch (err) {
        console.error(err);
      }
    }

    function updateReviewUI(status, notes) {
      const badge = document.getElementById('pageReviewBadge');
      document.getElementById('reviewNoteInput').value = notes || '';

      if (status === 'approved') {
        badge.className = 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800';
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>Approved</span>`;
      } else if (status === 'needs_review') {
        badge.className = 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-950 text-amber-300 border border-amber-800';
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span><span>Needs Review</span>`;
      } else if (status === 'flagged') {
        badge.className = 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-950 text-rose-300 border border-rose-800';
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span><span>Flagged Issue</span>`;
      } else {
        badge.className = 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700';
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-500"></span><span>Unreviewed</span>`;
      }
    }

    async function updateSummaryBadge() {
      try {
        const res = await fetch('/api/review-summary');
        const data = await res.json();
        const count = Object.keys(data.reviews || {}).length;
        document.getElementById('reviewedCountBadge').textContent = count;
      } catch (err) {}
    }

    async function toggleSummaryModal() {
      const overlay = document.getElementById('modalOverlay');
      document.getElementById('modalTitle').textContent = 'Review Progress & Verified Pages';
      const body = document.getElementById('modalBody');
      body.innerHTML = '<div class="text-xs text-slate-400 text-center py-4">Loading review summary...</div>';
      overlay.classList.remove('hidden');

      try {
        const [resRev, resAudit] = await Promise.all([
          fetch('/api/review-summary'),
          fetch('/api/audit')
        ]);
        const data = await resRev.json();
        const audit = await resAudit.json() || {};
        const reviews = data.reviews || {};
        const keys = Object.keys(reviews).sort((a,b) => parseInt(a) - parseInt(b));

        let approved = 0, flagged = 0, needsReview = 0;
        keys.forEach(k => {
          if (reviews[k].status === 'approved') approved++;
          else if (reviews[k].status === 'flagged') flagged++;
          else if (reviews[k].status === 'needs_review') needsReview++;
        });

        let auditBanner = '';
        if (audit.total_pages) {
          auditBanner = `
            <div class="bg-gradient-to-r from-blue-950/40 via-slate-900 to-emerald-950/40 border border-slate-800 p-3.5 rounded-xl mb-4">
              <div class="flex items-center justify-between text-xs font-semibold mb-2.5">
                <span class="text-emerald-400 font-bold flex items-center space-x-1.5">
                  <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                  <span>Full Scan Ink Coverage Audit (${audit.total_pages} Pages)</span>
                </span>
                <span class="text-emerald-300 font-mono text-[11px] bg-emerald-950/80 border border-emerald-800 px-2 py-0.5 rounded-full">${audit.clean_coverage_pct}% Clean</span>
              </div>
              <div class="grid grid-cols-4 gap-2 text-center text-xs">
                <div class="bg-slate-950/70 p-2 rounded-lg border border-slate-800">
                  <div class="text-base font-bold text-white">${audit.total_pages}</div>
                  <div class="text-[10px] text-slate-400">Total Pages</div>
                </div>
                <div class="bg-slate-950/70 p-2 rounded-lg border border-slate-800">
                  <div class="text-base font-bold text-blue-400">${audit.two_col_pages}</div>
                  <div class="text-[10px] text-slate-400">Two-Col Layout</div>
                </div>
                <div class="bg-slate-950/70 p-2 rounded-lg border border-slate-800">
                  <div class="text-base font-bold text-amber-400">${audit.total_pairs}</div>
                  <div class="text-[10px] text-slate-400">Verse-Trans Pairs</div>
                </div>
                <div class="bg-slate-950/70 p-2 rounded-lg border border-slate-800">
                  <div class="text-base font-bold ${audit.uncovered_anomalies === 0 ? 'text-emerald-400' : 'text-amber-400'}">${audit.uncovered_anomalies}</div>
                  <div class="text-[10px] text-slate-400">Ink Anomalies</div>
                </div>
              </div>
            </div>
          `;
        }

        let html = `
          ${auditBanner}
          <div class="grid grid-cols-3 gap-3 mb-4">
            <div class="bg-emerald-950/40 border border-emerald-800/60 p-3 rounded-lg text-center">
              <div class="text-2xl font-bold text-emerald-400">${approved}</div>
              <div class="text-xs text-emerald-300/80">Approved Pages</div>
            </div>
            <div class="bg-amber-950/40 border border-amber-800/60 p-3 rounded-lg text-center">
              <div class="text-2xl font-bold text-amber-400">${needsReview}</div>
              <div class="text-xs text-amber-300/80">Needs Review</div>
            </div>
            <div class="bg-rose-950/40 border border-rose-800/60 p-3 rounded-lg text-center">
              <div class="text-2xl font-bold text-rose-400">${flagged}</div>
              <div class="text-xs text-rose-300/80">Flagged Issues</div>
            </div>
          </div>
          <div class="border-t border-slate-800 pt-3">
            <h4 class="text-xs font-bold text-slate-300 mb-2">Reviewed Pages (${keys.length} / 530)</h4>
            <div class="space-y-2 max-h-72 overflow-y-auto">
        `;

        if (keys.length === 0) {
          html += `<div class="text-xs text-slate-500 italic text-center py-4">No pages reviewed yet. Review pages using the bottom bar!</div>`;
        } else {
          keys.forEach(k => {
            const r = reviews[k];
            let badgeClass = 'bg-slate-800 text-slate-400';
            if (r.status === 'approved') badgeClass = 'bg-emerald-950 text-emerald-300 border border-emerald-800';
            if (r.status === 'needs_review') badgeClass = 'bg-amber-950 text-amber-300 border border-amber-800';
            if (r.status === 'flagged') badgeClass = 'bg-rose-950 text-rose-300 border border-rose-800';

            html += `
              <div class="flex items-center justify-between p-2 rounded bg-slate-950 border border-slate-800 text-xs">
                <div class="flex items-center space-x-2">
                  <span class="font-bold text-white">Page ${k}</span>
                  <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${badgeClass}">${r.status}</span>
                  <span class="text-slate-400">${escapeHtml(r.notes || '')}</span>
                </div>
                <button onclick="changePage(${k}); closeModal()" class="text-emerald-400 hover:underline text-xs">Jump &rarr;</button>
              </div>
            `;
          });
        }

        html += `</div></div>`;
        body.innerHTML = html;
      } catch (err) {
        body.innerHTML = `<div class="text-xs text-rose-400">Failed to load review summary</div>`;
      }
    }

    async function exportReviewJson() {
      try {
        const res = await fetch('/api/review-summary');
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'santhya-vol-1-review.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      } catch (err) {
        alert('Export failed: ' + err);
      }
    }

    async function doSearch() {
      const q = document.getElementById('searchInput').value.trim();
      if (!q) return;

      const overlay = document.getElementById('modalOverlay');
      document.getElementById('modalTitle').textContent = `Search Results: "${q}"`;
      const body = document.getElementById('modalBody');
      body.innerHTML = '<div class="text-xs text-slate-400 text-center py-4">Searching across 530 pages...</div>';
      overlay.classList.remove('hidden');

      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        const results = data.results || [];

        let html = `
          <div class="text-xs text-slate-400 mb-3">Found <strong>${results.length}</strong> matching lines across Volume 1:</div>
          <div class="space-y-2 max-h-96 overflow-y-auto">
        `;

        if (results.length === 0) {
          html += `<div class="text-xs text-slate-500 italic text-center py-6">No matching text found.</div>`;
        } else {
          results.forEach(r => {
            html += `
              <div class="p-2.5 rounded bg-slate-950 border border-slate-800 hover:border-slate-700 transition cursor-pointer" onclick="changePage(${r.page}); closeModal()">
                <div class="flex items-center justify-between text-xs mb-1">
                  <div class="flex items-center space-x-2">
                    <span class="text-emerald-400 font-bold">Page ${r.page}</span>
                    <span class="text-slate-500 font-mono text-[11px]">Line #${r.line_n}</span>
                    <span class="px-1.5 rounded bg-slate-800 text-slate-400 text-[10px]">Col ${r.col !== null ? r.col : 'Single'}</span>
                  </div>
                  <span class="text-slate-400 text-xs hover:underline text-emerald-400">Jump to Page &rarr;</span>
                </div>
                <div class="gurmukhi text-sm text-slate-200">${highlightMatch(escapeHtml(r.text), q)}</div>
              </div>
            `;
          });
        }

        html += `</div>`;
        body.innerHTML = html;
      } catch (err) {
        body.innerHTML = `<div class="text-xs text-rose-400">Search error.</div>`;
      }
    }

    function highlightMatch(text, query) {
      if (!query) return text;
      const parts = text.split(query);
      return parts.join(`<mark class="bg-amber-500/30 text-amber-200 px-0.5 rounded">${query}</mark>`);
    }

    function closeModal() {
      document.getElementById('modalOverlay').classList.add('hidden');
    }

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Deep linking & history support
    window.addEventListener('hashchange', () => {
      const h = window.location.hash.replace('#', '');
      if (h && !isNaN(parseInt(h)) && parseInt(h) !== currentPage) {
        changePage(parseInt(h));
      }
    });

    // Startup
    window.onload = () => {
      let initialPage = 201;
      const hash = window.location.hash.replace('#', '');
      if (hash && !isNaN(parseInt(hash))) {
        initialPage = parseInt(hash);
      } else {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('page') && !isNaN(parseInt(urlParams.get('page')))) {
          initialPage = parseInt(urlParams.get('page'));
        }
      }
      changePage(initialPage);
      updateSummaryBadge();
    };
  </script>
</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/page/") and path.endswith("/image"):
            parts = path.strip("/").split("/")
            try:
                page_num = int(parts[2])
                img_path = os.path.join(PAGES_DIR, f"{page_num:04d}.png")
                if os.path.exists(img_path):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(os.path.getsize(img_path)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    return
            except Exception:
                pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/stats":
            eval_data = {}
            if os.path.exists(EVAL_FILE):
                try:
                    with open(EVAL_FILE, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)
                except Exception:
                    pass
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "book": "santhya-vol-1",
                "title": "Santhya Sri Guru Granth Sahib Ji, Vol. 1",
                "author": "Bhai Vir Singh",
                "pages": 530,
                "engines": eval_data.get("engines", []),
            }).encode("utf-8"))
            return

        if path.startswith("/api/page/") and path.endswith("/image"):
            # /api/page/<num>/image
            parts = path.strip("/").split("/")
            try:
                page_num = int(parts[2])
                img_path = os.path.join(PAGES_DIR, f"{page_num:04d}.png")
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        data = f.read()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception as e:
                print(f"[viewer] Error serving image: {e}")
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        if path.startswith("/api/page/"):
            # /api/page/<num>
            try:
                page_num = int(path.strip("/").split("/")[2])
                data = get_page_data(page_num)
                if data:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
                    return
            except Exception as e:
                print(f"[viewer] Error serving page data: {e}")
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        if path == "/api/search":
            qs = urllib.parse.parse_qs(parsed.query)
            q = qs.get("q", [""])[0].strip()
            results = []
            if q:
                # search in SEARCH_INDEX
                for pg, line_n, col, txt in SEARCH_INDEX:
                    if q in txt:
                        results.append({
                            "page": pg,
                            "line_n": line_n,
                            "col": col,
                            "text": txt,
                        })
                        if len(results) >= 100:
                            break
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"results": results}, ensure_ascii=False).encode("utf-8"))
            return

        if path == "/api/audit":
            audit_file = os.path.join(BASE_DIR, "data", "raw", "ocr-audit-santhya-vol-1.json")
            audit_data = {}
            if os.path.exists(audit_file):
                try:
                    with open(audit_file, "r", encoding="utf-8") as f:
                        audit_data = json.load(f)
                except Exception as e:
                    print(f"[viewer] Error loading audit file: {e}")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(audit_data, ensure_ascii=False).encode("utf-8"))
            return

        if path == "/api/review-summary":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reviews": REVIEW_STATUS}, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/page/") and path.endswith("/review"):
            try:
                page_num = str(int(path.strip("/").split("/")[2]))
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                
                REVIEW_STATUS[page_num] = {
                    "status": payload.get("status", "unreviewed"),
                    "notes": payload.get("notes", ""),
                }
                save_review_status()

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
                return
            except Exception as e:
                print(f"[viewer] Error saving review: {e}")
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.end_headers()
                return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()


def run_server(port=8765):
    init_data()
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, RequestHandler)
    print(f"\n=======================================================")
    print(f" Punjabi OCR Review Viewer Running at:")
    print(f" 👉 http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] Shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
