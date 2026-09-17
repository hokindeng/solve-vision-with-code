# Visual style contract — paper figures and results site

Owner: CTO. Every figure script (`make_*.py`), every table caption and the site (`docs/`) follow this file.
Reference look: `object-permanence-paper/2026-09-13_overleaf-export/build/main.pdf` — figures sit next to
the paragraph that discusses them, show real task frames, and use grey/black with one accent.

## 1. Measured template facts (do not re-derive)

| Fact | Value | Source |
|---|---|---|
| Template mode | `\usepackage[pr]{icml2026}` → **one column** | `main.tex:68` |
| `\textwidth` | **6.0 in** (`\columnwidth` = `\textwidth` in this mode) | `icml2026.sty:825` |
| `\textheight` | 8.7 in | `icml2026.sty:826` |
| Body font | Times (`\RequirePackage{times}`), 10 pt; captions 9 pt | `icml2026.sty:116` |
| Float packages loaded | `float` (`[H]` works), `wrapfig`, `subcaption`, `caption`, `longtable`. **`placeins` is NOT installed** (BasicTeX; `kpsewhich placeins.sty` empty) | `main.tex:11–39`, kpsewhich |
| Template greys | `pr@lightgray`, `pr@darkgray` (header/footer rules) | `icml2026.sty` |

Consequence: the old `SINGLE, DOUBLE = 3.25, 6.75` in `make_figures.py` is wrong for this template. A 3.25 in
figure placed at `\columnwidth` is blown up 1.85×; a 6.75 in figure is shrunk. Export at the final printed
width and include with the matching fraction of `\textwidth`, so text inside a figure is never rescaled.

## 2. Figure widths (inches) and the matching `\includegraphics`

| Name | Width | Use | LaTeX |
|---|---|---|---|
| `FULL` | 6.0 | heatmaps, qualitative grids, leaderboard | `width=\textwidth` |
| `WIDE` | 5.4 | teaser (reference paper uses 0.9) | `width=0.9\textwidth` |
| `HALF` | 2.9 | one of a side-by-side pair | `width=0.48\textwidth` inside two `subfigure`s / minipages |
| `WRAP` | 2.7 | `wrapfigure{r}{0.47\textwidth}` for a small chart beside prose | `width=\linewidth` |

Heights: keep a full-width figure ≤ 3.9 in (0.45 `\textheight`) so `[H]` never pushes a half-empty page.
Teaser ≤ 2.6 in. Save with `bbox_inches="tight", pad_inches=0.02`; then the printed width is what you asked for
minus at most a hairline.

## 3. Palette (print)

| Token | Hex | Role |
|---|---|---|
| `INK` | `#111111` | all text, spines, annotation |
| `GREY_1` | `#555555` | secondary text: axis labels, lane labels, sub-captions |
| `GREY_2` | `#999999` | tertiary: tick labels on dense axes, reference lines, "n=" notes |
| `GREY_3` | `#dddddd` | thumbnail borders, table rules, light gridlines |
| `PAPER` | `#ffffff` | background; never tint panels |
| `AGENT` | `#2a5db0` | **the one accent**: coding agents (closed lanes codex / claude / gemini) |
| `AGENT_OPEN` | `#a9c0e6` | open-weight coding agents (a 45 % tint of `AGENT`; same hue, so "agents" reads as one family) |
| `VIDEO` | `#a8a29a` | video models: muted warm grey, no edge line |
| `MARK_OK` | `#2a5db0` | a score ≥ 0.9 printed in accent on a badge (optional) — never green/red traffic lights |

Rules: no colormaps other than a single-hue ramp `#ffffff → #2a5db0` (heatmaps). Never use matplotlib
defaults (`C0`…), never `tab10`, never `viridis` in this paper. Grayscale-safe: agents are dark, open-weight
agents are light-blue, video models are mid-grey — the three separate when printed in black and white.

Category (Abstraction / Perception / Spatiality / Transformation / Knowledge) is **not** a colour axis.
Show categories by position (columns, facets) and label them in `GREY_1` text.

## 4. matplotlib rcParams — paste verbatim at the top of every `make_*.py`

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GREY_1, GREY_2, GREY_3 = "#111111", "#555555", "#999999", "#dddddd"
AGENT, AGENT_OPEN, VIDEO = "#2a5db0", "#a9c0e6", "#a8a29a"
FULL, WIDE, HALF, WRAP = 6.0, 5.4, 2.9, 2.7          # inches, = 1.0 / 0.9 / 0.48 / 0.45 \textwidth

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],   # verified present on this Mac
    "mathtext.fontset": "stix",                 # Times-like math
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 4, "patch.linewidth": 0.6,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType so Times survives in the PDF
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
```

Verify once: `python3 -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if f.name.startswith('Times')])"`
prints `Times New Roman` (checked 2026-09-16; matplotlib 3.11.1).

## 5. Marks

- **Bars**: fill by family (`AGENT` / `AGENT_OPEN` / `VIDEO`), no edge line, height 0.72 of the slot. Sort by the
  value plotted, never alphabetically. Print the value at the bar end in 7 pt `INK`, 3 decimals.
- **Dumbbells** (in-domain vs out-of-domain): connector `GREY_3` 0.8 pt; in-domain dot hollow (`PAPER` fill,
  family edge 0.8 pt), out-of-domain dot solid family fill, size 4.
- **Lines**: 1.2 pt; at most 3 lines per axes; label the line at its right end in the line's colour, no legend box.
- **Heatmap**: single-hue ramp white→`AGENT`, cell text in `INK` when value < 0.6 else `PAPER`; cell borders
  `PAPER` 0.5 pt; no colorbar if every cell carries its number.
- **Reference lines** (chance, best video model): `GREY_2` dashed `(0,(3,2))` 0.6 pt, labelled in 7 pt `GREY_2`.
- **Gridlines**: only horizontal, only on bar charts taller than 2 in, `GREY_3` 0.4 pt.
- **Legend**: only when family colours appear; place inside the axes top-left or below the plot; text `GREY_1`.
- **Titles**: none inside the PDF. The caption carries the title. Panel letters `(a)`, `(b)` in 8 pt bold `INK`,
  top-left, outside the axes.

## 6. Numbers

- Scores print with **three decimals**: `f"{x:.3f}"` → `0.923`. No percent signs, no leading-zero stripping,
  no rounding to two places even in tight cells. Counts are integers with no thousands separator below 10 000.
- **Every number is read from a file, never typed.** Sources of record: `bench/paper/table1_leaderboard.csv`,
  `table2_categories.csv`, `table3_efficiency.csv`, `table4_per_task_closed.csv`, `bench/stats.json`, and per-instance
  scores from `~/Workspace/svc-media/index.json` (which mirrors `bench/results/`). A script with a literal score
  in it fails review. Every `make_*.py` has `--check`, which recomputes what it draws and exits non-zero on a
  mismatch with the CSV; `paper/check_numbers.py` (SWE-4) cross-checks the `.tex` prose against the same CSVs.
- Model display names come from one place: `bench/paper_tables.py::short` (do not add another alias table).

## 7. Real imagery: frames, thumbnails, badges

Figures that explain a task show **actual frames** (`~/Workspace/svc-media/index.json` → `gt.first_frame`,
`gt.final_frame`, `lanes.<lane>.last_frame`, all 1024×1024 RGB PNG). No schematic boxes, no clip-art arrows,
no fake "code editor" rectangles.

- **Thumbnail**: square, drawn with `ax.imshow` on an axes with all spines visible at 0.5 pt `GREY_3`, ticks off,
  no shadow, no rounded corners, no drop-shadow. Gap between thumbnails 0.06 in.
- **Row/column labels**: 7 pt `GREY_1`, small caps via `.upper()` is fine for lane names (`CODEX · CLAUDE · GEMINI`);
  the GT column is labelled `ground truth`. Prompt text, when shown, is 7 pt `INK`, wrapped at ~60 chars, max 3 lines,
  ellipsis after that.
- **Score badge**: a rectangle inside the thumbnail, bottom-right, 0.36 × 0.16 in, `PAPER` fill at alpha 0.92,
  `GREY_3` 0.5 pt edge, text `f"{score:.3f}"` 7 pt `INK`. Use `ax.text(0.97, 0.04, ..., transform=ax.transAxes,
  ha="right", va="bottom", bbox=dict(boxstyle="square,pad=0.25", fc="white", ec=GREY_3, lw=0.5, alpha=0.92))`.
  Do not colour badges by pass/fail.
- **Code panels** (`fig_code_*.pdf`): monospace `DejaVu Sans Mono` 6.5 pt `INK` on `PAPER`, at most 22 lines, a
  `GREY_3` 0.5 pt frame; the file is the agent's real `solve.py`, trimmed to the lines the caption discusses, with
  `…` on its own line where cut. No syntax highlighting.
- **Teaser** (`fig_teaser.pdf`, WIDE): left GT first frame + prompt, middle the last frame produced by the best lane
  with its badge, right the GT final frame; a single thin `GREY_1` arrow between panels at most. One line of 7 pt
  `GREY_1` below: task id, category, lane. Nothing else.

## 8. Float placement (Lead 1)

- Default `\begin{figure}[H]` / `\begin{table}[H]` placed **immediately after the paragraph that first cites the
  float**, and keep each float under 0.45 `\textheight` so `[H]` cannot leave a large gap. `figure*` = `figure` in
  this one-column mode; the star does nothing, drop it except in the teaser where the reference paper keeps it.
- Teaser: `\begin{figure*}[h]` exactly as the reference paper's abstract file.
- Small charts (`fig_ood`, `fig_tools_vs_score`): `wrapfigure{r}{0.47\textwidth}` beside the prose, `\vspace{-8pt}` top
  and bottom as in the reference `3_dataset.tex`. Never start a wrapfigure within 6 lines of a section heading or a list.
- `[t]` only when a float is cited in the first three lines of a page. Never `[htbp]` here (that is how everything drifted
  to the end).
- `\clearpage` before the appendix (already in `main.tex:219/230`). Table 7 (`tab_tasks.tex`, `longtable`) stays in the appendix
  after that `\clearpage`; longtable is not a float and needs `\endfirsthead`/`\endhead` for its repeated header.
- Fix the existing "Float too large for page by 12.13pt" (`main.log:1126`, `5_analysis.tex:56`) by shrinking that figure's
  height, not by `\resizebox`.

## 9. Prose (Lead 1, checked by SWE-4 `lint_prose.py`)

Flag and rewrite: em-dash chains (more than one `---` in a sentence), "not X but Y", triads of parallel adjectives,
a paragraph that ends with a one-sentence summary of itself, "delve", "landscape", "crucially", "notably", "it is worth noting",
"serves as", "underscores", "highlights the importance". Sentences ≤ 28 words on average. Numbers in prose match the CSV.

## 10. Site tokens (Lead 2 `docs/style.css`) — same semantics, dark theme of object-permanence.world

```css
:root {
  --bg: #000000;          /* page */
  --card: #111111;        /* cards, table rows */
  --card-2: #1c1c1e;      /* hover / nested panel */
  --border: #2a2a2c;      /* thumbnail frame, hairlines (print GREY_3) */
  --text: #f5f5f7;        /* print INK */
  --text-2: #a1a1a6;      /* print GREY_1 */
  --text-3: #6e6e73;      /* print GREY_2 */
  --agent: #2997ff;       /* accent: closed coding agents */
  --agent-open: #7fb8ff;  /* open-weight coding agents (tint) */
  --video: #8e8e93;       /* video models */
  --heat-0: #111111; --heat-1: #2997ff;   /* single-hue ramp for heatmaps */
  --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --radius: 12px; --gap: 16px; --maxw: 1200px;
}
```

- Inter from Google Fonts (`https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap`), weights 400/500/600 only.
- Thumbnails: 1 px `--border`, `border-radius: 6px`, no shadow. Score badge: bottom-right, `--card` at 0.85 alpha, `--text`, 12 px, tabular numerals (`font-variant-numeric: tabular-nums`), three decimals.
- Bars on the site use the same three family colours; category is again position, not colour.
- No company name, logo or link anywhere in `docs/`; author line is Hokin Deng, link hokindeng.com and the GitHub repo.

## 11. Site data contract (SWE-3 writes, Lead 2 reads)

**The contract is the module docstring of `docs/build_site.py`** (SWE-3 owns it; Lead 2 codes against it; the CTO's
earlier draft schema in this section is withdrawn so there is one text, not two). Invariants this file adds on top:

- Every path in `docs/data/*.json` is relative to `docs/`, no leading slash, so the site works both at
  `https://hokindeng.github.io/solve-vision-with-code/` and under `cd docs && python3 -m http.server`.
- `score` is a JSON number, never a string; the site prints `score.toFixed(3)`; a lane with no produced video has `video: null`,
  `last_frame: null` and its recorded score (0.0).
- Model display names and lane order come from `bench/paper_tables.py` (`short`, `CLOSED`) — the builder imports them; nothing is retyped.
- `tasks.json` has exactly 100 tasks × 5 instances, closed lanes `codex claude gemini` present on every instance.
- `solve.py` text is never inlined in JSON; one file per closed instance under `data/solve/<lane>/<short>/<idx>.py`
  (1 500 files, 4.2 MB), fetched on expand with `fetch(url).then(r => r.text())` into a `<pre class="code">`.
- Media naming: `media/<lane>/<split>/<short>/<idx>.mp4|.jpg`, GT adds `_first.jpg`; `<split>` and `<lane>` slugs are whatever
  the docstring says. Lead 2 never builds a path by hand — every URL comes out of the JSON — and normalises `split` once on load
  (`/^In-Domain/` → In-Domain, `/^Out-of-Domain/` → Out-of-Domain) so either spelling of the split works.
- Video encoding the builder guarantees for **every** shipped mp4, copied or re-encoded: `h264`, `yuv420p`, `+faststart`
  (601 source files are MPEG-4 Part 2 — **all 500 GT `ground_truth.mp4`**, 96 deepseek, 5 kimi — plus 9 gbrp, 6 yuv444p,
  4 yuvj420p incl. one claude file; none play in Safari/Chrome; and every source has its moov atom after mdat).
  A copied-through file must at least be remuxed: `ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4`.
- Thumbnails are JPEG (≤ 384 px, q 82–85), never the 1024² PNGs.
- `<video muted playsinline loop preload="none" poster="<last_frame>">`; play on hover / IntersectionObserver, ≤ 12 playing at once.

## 12. Size budget for `docs/` (GitHub Pages, branch `public`, path `/docs`)

| Asset | Source size | Shipped |
|---|---|---|
| Closed-lane videos 1 500 | 75 MB | ≈ 60 MB after re-encode |
| GT videos 500 | 144 MB | ≈ 90 MB (crf 26); acceptable — or ship GT first/last frames only and link the HF dataset if the total passes 300 MB |
| Open-lane videos 1 468 | 207 MB | **not all**: ship at most 2 instances per task per open lane (≤ 600 files, ≈ 50 MB) as `open_examples` |
| Thumbnails 3 468 + GT 1 000 | 153 MB PNG | ≈ 80 MB as 384 px JPEG |
| solve.py 1 500 | 4.2 MB | as is |
| **Total** | | **target ≤ 300 MB, hard stop 500 MB** |

Hard limits: any single file < 100 MB (git push), repo soft limit 1 GB, Pages site soft limit 1 GB, Pages bandwidth 100 GB/month.
**Never put `docs/media` in Git LFS** — Pages serves LFS pointers, not payload. Add `docs/.nojekyll`.
