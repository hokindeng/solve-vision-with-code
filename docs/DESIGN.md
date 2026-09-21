# DESIGN — solve-vision-with-code.com (Site B), 2026-09-20

Design contract for the redesign of `docs/` in the object-permanence.world shape, done at Hokin's request.
Seats: Lead 2 (`index.html`, `leaderboard.html`, `style.css`, `build_site.py`), SWE-4 (`app.js`, `leaderboard.js`),
SWE-5 (`verify-site.py`, `DEPLOY.md`). Reference: `/Users/hokin/hokin/project/object-permanence.world/app.py`.
The tokens, hero, tile, panel and scroll-bar rules are the same as Site A's
`/Users/hokin/hokin/project/code-world-model-bench.com/DESIGN.md` §1–4; this file states only what differs and
what is specific to this site. Where this file and `BRIEF-SITES.md` disagree, this file wins.

## 0 What must not change

- Every number. `docs/data/leaderboard.json`, `docs/data/tasks.json`, `docs/data/solve/**`, `docs/media/**`,
  `docs/paper.pdf` are read-only for this redesign; `build_site.py`'s self-check stays green (`--no-media` run).
- Model display names: exactly the strings in `docs/data/leaderboard.json` (`Codex/gpt-6-astra`,
  `Claude Code/Fable 5.1`, `Gemini CLI/3.1 Pro`, `GLM-5`, …, `VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)`)
  and, for lane headers, `tasks.json.lanes[<lane>].label` (`Codex / gpt-6-astra`, `Claude Code / Fable 5.1`,
  `Gemini CLI / 3.1 Pro`, `GLM-5 (OpenCode)`, …). No renaming, no "Claude Opus" — this paper ran Fable 5.1 and says so.
- Public repo hygiene: commits carry no `Co-Authored-By` trailer; the commit message says the redesign was done at
  Hokin's request; only `docs/` and one README line change.
- Paths stay relative (no leading `/`): the same files serve from `https://solve-vision-with-code.com/` and from
  `https://hokindeng.github.io/solve-vision-with-code/`.
- Attribution is the author's only: no organisation name, organisation e-mail or organisation repository link
  anywhere in `docs/`. `docs/verify-site.py` greps for the list it carries (the list itself lives in that script and
  in the private build notes, not in this public file).

## 1 Tokens

`:root` identical to Site A §1 (`--bg #000 · --card #111 · --card-hover #171717 · --border rgba(255,255,255,.08)
· --text #f5f5f7 · --t2 .56 · --t3 .34 · --accent #2997ff · --accent-dim rgba(41,151,255,.14)`), Inter via the
Google Fonts `<link>` already in `index.html`, `.grad` gradient identical. Site B keeps its extra tokens because the
leaderboard needs three kinds: `--open:#5ac8fa`, `--video:#8e8e93` (+ `-dim`), `--border-strong:rgba(255,255,255,.16)`,
`--mono`. The split colour for OOD stays `#ffb340` (only on the tile badge and the panel pill). Nothing else is
coloured.

## 2 Hero (both pages, verbatim apart from `.cur`)

```html
<header class="hero" id="top">
 <h1 class="grad">Solve Vision with Code</h1>
 <h2>Coding Agents Are Stronger Visual Reasoners Than Video Models</h2>
 <p class="lead">A coding agent is given the first frame and the prompt of a VBVR-Pro-Bench task and writes a program that renders the answer video; the video is scored by the benchmark's official evaluator, so it sits on the same leaderboard as the video generation models.</p>
 <div class="bignums" id="bignums">
  <div class="bn agent"><div class="v" id="bn-agent">—</div><div class="l">best coding agent</div><div class="m" id="bn-agent-m"></div></div>
  <div class="vs">VS</div>
  <div class="bn video"><div class="v" id="bn-video">—</div><div class="l">best video model</div><div class="m" id="bn-video-m"></div></div>
 </div>
 <div class="hero-links">
  <a class="hbtn" href="paper.pdf" target="_blank" rel="noopener">Paper</a>
  <a class="hbtn" href="https://github.com/hokindeng/solve-vision-with-code" target="_blank" rel="noopener">Code</a>
  <a class="hbtn" href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-Bench" target="_blank" rel="noopener">Benchmark</a>
  <a class="hbtn" href="leaderboard.html">Leaderboard</a>
  <a class="hbtn" href="#cite">Cite</a>
 </div>
 <div class="hero-rule"></div>
</header>
```

- `h1`: `clamp(26px,3.6vw,42px)`/800, `-.03em`, `line-height:1.12`, `margin-bottom:10px` (a step above Site A's
  `clamp(22px,3vw,36px)` because an `h2` sits under it; one rule in `style.css`, so both pages match). `h2`: `clamp(15px,2vw,20px)`/600 `var(--t2)`
  `-.015em`, `margin-bottom:16px`. `.lead`: 15px `var(--t2)`, `max-width:720px`, `margin:0 auto 26px`.
- The two big numbers stay (the paper's one-line result): `.bignums` as today, filled by JS from `leaderboard.json`
  (`best overall` among `kind` containing "coding agent" but not "open", and among "video model"), three decimals,
  model string verbatim under each. On `leaderboard.html` the same block appears (same JS).
- No nav bar, no `primary` button, no `hokindeng.com` button (it moves to the footer). `.hbtn.cur` on the current page's
  Leaderboard button. On `leaderboard.html` the Cite button is `index.html#cite`.
- The `.hero-rule` is the last element of the hero. Straight into the filter row on `/`, into the table on the
  leaderboard page.

## 3 Home page `index.html`

### 3.1 Filter pills (ported from today's `renderGallery.apply`)

Under the rule, one centred row `.filters` (`gap:8px;flex-wrap:wrap;margin:22px auto 0;max-width:1400px;padding:0 24px`):

- Split group (single-select): `All 100` · `In-Domain 50` · `Out-of-Domain 50` (`.rpill`, `.on` = accent look).
- Category group (single-select, `All` default): `Abstraction 25 · Perception 28 · Spatiality 16 · Transformation 12 · Knowledge 19`
  — counts computed from `tasks.json`, never typed.
- Search input `.fsearch` (same pill box, `min-width:220px`, placeholder `Search id, name or prompt…`), debounced 120 ms.
- Count at the row's end: `100 / 100 tasks` (`var(--t3)`).
- State is mirrored to the URL hash as `#split=In-Domain&cat=Perception&q=…` so a filtered view is linkable;
  `#task-<id>` still deep-links a task (opens its panel). Filtering closes an open panel.

### 3.2 Grid and tile

`.grid` as Site A but `grid-template-columns:repeat(auto-fill,minmax(200px,1fr))` and **tiles are square**
(`aspect-ratio:1/1`): every video on this benchmark is 1024² and a 16:9 crop would cut the top or bottom of the
scene (the sequence row and the choices row are what the task is about). The panel cells are square too. Everything
else about `.tile` (radius 12, border, hover lift, `.sel`) is Site A's.

```html
<div class="tile" data-id="G-131_select_next_figure_increasing_size_sequence_data-generator">
 <video data-src="media/gt/In-Domain_50/G-131_select_next_figure_increasing_size_sequence/00000.mp4" poster="media/gt/In-Domain_50/G-131_select_next_figure_increasing_size_sequence/00000_last.jpg" muted loop playsinline preload="none"></video>
 <span class="tile-badge split-id">ID</span>
 <span class="tile-badge cat">Abstraction</span>
 <div class="tile-name">G-131 · select next figure increasing size sequence</div>
</div>
```

- Tile video = the ground truth of instance 0 (`tasks[i].gt.video`, poster `gt.last_frame`, so the grid paints
  before any video loads). Lazy `data-src` + IntersectionObserver exactly as Site A.
- Two badges: top-left the split (`ID` accent / `OOD` `#ffb340`, `.tile-badge` box from Site A), top-right the
  category in `var(--t2)`. Hover overlay `.tile-name` = `<G-131> · <short name with underscores → spaces>`.
- Order = `tasks.json` order (In-Domain first). No sorting controls.

### 3.3 Inline panel

Reference mechanics verbatim (`toggleTask` / `rowEnd` / `closePanel` / `initScroller` / `replayAll` / `toggleAll`,
Escape, resize re-insert, one panel at a time). SWE-4 ports them from `app.py` lines 312–388 into `app.js`.

Header `.dtop`: `.dtitle` = `G-131_select_next_figure_increasing_size_sequence` (the `short` id, monospace 13px is
fine) + `.pill split-id` `In-Domain` + `.pill` `Abstraction` · spacer · `⟲ Replay all` · `❚❚ Pause all` · `✕`.

Prompt: `.mprompt` (reference: 14px `var(--t2)` centred, `max-width:960px`) showing `tasks[i].prompt`, clamped to
four lines with a `Full prompt` / `Less` toggle when longer than 420 characters (today's logic).

Top scroll bar `.sbar` (white thumb, `Scroll right →`), then `.cmpwrap > .cmp`:

```
grid-template-columns: 96px repeat(4, minmax(220px,1fr)); gap:10px; align-items:start
```

Columns, fixed order: 1 `codex` · 2 `claude` · 3 `gemini` · 4 Ground truth (`.chead.ref`, accent). Header cell
= `lanes[l].label.split(' / ')`: line 1 the agent (`Codex`, `Claude Code`, `Gemini CLI`), `small` the model
(`gpt-6-astra`, `Fable 5.1`, `3.1 Pro`). Ground truth's `small` = `official video`.

Rows = the five instances `00000…00004`. Sticky first cell `.crow.cst` holds the instance's **first frame**
(`instances[k].gt.first_frame`, 72×72, radius 6) over the label `instance 0` — the input the agent saw, so each
row reads input → three answers → truth.

Cell (agent lane): square video (`instances[k].lanes[l].video`, poster `last_frame`, `controls`, lazy) and a
`.score` line `<b>0.928</b> · <button class="plink">program</button>`. Score three decimals; `1.000` prints in
`var(--accent)`, everything else `var(--text)`. Missing video (`video` null or load error) → `.vph` `no video` and
`.score` `<b>0.000</b>` (an instance without a video scores 0 by the official rule). Ground-truth cell: video +
`.score` `ground truth` muted.

`program` toggles a `.code` block (today's markup: `.ch` header with `Codex · data/solve/codex/…/00000.py` + `raw`
link, `<pre>` with the fetched text) inserted after the table inside the panel; one open at a time; clicking the
same button closes it. Keep the fetch cache.

Under the table (and under an open code block): a collapsed section in the reference's `.sec.tog` style,
`▸ Open-weight examples <span>N videos · GLM-5, Kimi K2.5, DeepSeek V3.2 through OpenCode</span>`, closed by
default. Opening it renders this task's `open_examples` (≤ 6: two best instances per open lane) as `.tgrid` cards
(`minmax(200px,1fr)`): square video (poster `last_frame`, lazy), caption `<lane label without " (OpenCode)"> · instance k · 0.xxx`.
Hidden entirely when the task has no `open_examples` with a video. This replaces both the old section 06 sample of
12 cards and the old per-card "open-weight attempts" button; the home page has no other section.

### 3.4 After the grid

Nothing between the grid and the footer. The heatmap (old section 04) is retired: the grid + panel is the per-task
view now. `tasks[i].mean` stays in the data unused.

## 4 Leaderboard page `leaderboard.html` (new; `leaderboard.js`, SWE-4)

Hero (§2, Leaderboard `.cur`), then `.wrap{max-width:1100px;margin:0 auto;padding:30px 24px 90px}`:

1. `.sec` `Leaderboard <span>VBVR-Pro-Bench, video setting · 100 tasks × 5 instances · official evaluator</span>`
   and the existing `.sub` sentence ("Mean official-evaluator score over all 500 instances; an instance without a
   video counts 0. Video-model rows are the published VBVR-Pro-Bench numbers. Click a header to sort."), legend
   (coding agent · open weights · video model) as today.
2. The sortable 37-row table exactly as today (`Rank · Model · Agent · Overall · ID · OOD · Videos`, dots, bars,
   `hide-m` columns), moved verbatim from `app.js` into `leaderboard.js` (`prepareLeaderboard`, `renderLeaderboard`,
   `CLOSED_AGENT` versions map unchanged).
3. `.sec` `Out-of-domain vs in-domain` + the `.sub` sentence + the dumbbell SVG (`renderDumbbell`, moved verbatim;
   it is inline SVG drawn without a library and stays JS-drawn because it is responsive to the card width).
4. `.sec` `Categories & efficiency` + `.sub` + the two small tables with their `Show all agents` buttons
   (`renderCategories`, `renderEfficiency`, moved verbatim).
5. Footer (§5).

`leaderboard.js` loads only `data/leaderboard.json`; it does not load `tasks.json`. Shared helpers (`el`, `svg`,
`f3`, `kindOf`, big-number fill, lazy video, scroller) live in `app.js` only if `app.js` is split into a shared part;
otherwise duplicate the ~40 lines rather than load `tasks.json` on the leaderboard page. Lead 2 decides and writes it
in the file header; SWE-4 follows.

## 5 Footer, cite, `<head>`

Footer (both pages, `id="cite"` on the home page; the leaderboard page's footer has no bib block, only the two lines):

```html
<footer id="cite">
 <h4>Cite</h4>
 <div class="bib" id="bib"><button class="copy" type="button" id="copybib">Copy</button>@article{deng2026solvevision,
  title   = {Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models},
  author  = {Deng, Hokin and Zhao, Zehong and Ji, Ran and Wang, Maijunxian and Gao, Qingying and Yu, Fengyuan and Jiang, Zhengze and Zhang, Yilan},
  year    = {2026},
  url     = {https://solve-vision-with-code.com/}
}</div>
 <p>Code, results and this page are released for research and non-commercial use only; see the <a href="https://github.com/hokindeng/solve-vision-with-code/blob/public/LICENSE">LICENSE</a>. VBVR-Pro-Bench and its evaluator belong to their authors.</p>
 <p>Hokin Deng · <a href="https://hokindeng.com">hokindeng.com</a> · <a href="https://github.com/hokindeng/solve-vision-with-code">GitHub</a></p>
</footer>
```

`<head>` (home page; the leaderboard page changes title to `Leaderboard — Solve Vision with Code`, canonical and
`og:url` to `https://solve-vision-with-code.com/leaderboard.html`):

```html
<title>Solve Vision with Code</title>
<meta name="description" content="Coding Agents Are Stronger Visual Reasoners Than Video Models — results, leaderboard and every generated video on VBVR-Pro-Bench.">
<meta name="author" content="Hokin Deng">
<meta property="og:title" content="Solve Vision with Code">
<meta property="og:description" content="Coding Agents Are Stronger Visual Reasoners Than Video Models. A coding agent reads the first frame and the prompt of a VBVR-Pro-Bench task and writes a program that renders the answer video.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://solve-vision-with-code.com/">
<link rel="canonical" href="https://solve-vision-with-code.com/">
<meta name="color-scheme" content="dark"><meta name="theme-color" content="#000000">
```

Favicon: keep today's inline SVG (`</>` glyph in accent on a black rounded square). Fonts `<link>` as today.
The author list is the paper's; it is not filtered here (the Departed rule applies to new author lists, and this
one is the published paper's).

## 6 Mobile

Site A §9 applies: 16 px gutters ≤ 640 px, no horizontal page scroll, panel table scrolls inside `.cmpwrap`,
leaderboard tables inside `.tscroll`. Square tiles at `minmax(200px,1fr)` give one column at 375 px; `.cmp` columns
`minmax(180px,1fr)` ≤ 640 px with the sticky first column at 64 px (first frame 48×48). `.bignums` stack as today.
Filter pills wrap; the search input takes the full width ≤ 640 px.

## 7 Files, build, verify

- `docs/index.html`, `docs/leaderboard.html`, `docs/style.css` (Lead 2); `docs/app.js`, `docs/leaderboard.js`
  (SWE-4); `docs/verify-site.py`, `docs/DEPLOY.md` (SWE-5); this file (CTO). The old `index.html` / `style.css` /
  `app.js` are moved to `~/hokin/archive/2026-09/2026-09-20_svc-docs-v1/` by Lead 2 before overwriting (Rule 3;
  they remain in git history too).
- `docs/build_site.py`: `self_check` additionally asserts `docs/leaderboard.html` exists and that `index.html`,
  `leaderboard.html`, `app.js`, `leaderboard.js`, `style.css` contain none of the banned strings that
  `docs/verify-site.py` defines (import its list; do not copy it) and that both HTML files carry `canonical` = the
  domain. Nothing else changes in it.
- `docs/verify-site.py` (SWE-5): the banned-string grep over `docs/*.html|js|css|md` (organisation name and e-mail,
  organisation repository prefix, bucket URLs, cloud host names, IPv4 addresses — the same list as Site A's §13 minus
  the Site A-only model and cell names), every relative `href`/`src`
  in the two HTML files resolves under `docs/`, `data/leaderboard.json` has 37 rows with top overall 0.923,
  `data/tasks.json` has 100 tasks × 5 instances, a 40-file random sample of referenced media exists, and no
  `Co-Authored-By` in `git log -5 --format=%B` of `docs/`.
- Deploy (`docs/DEPLOY.md`, SWE-5; parent runs, Hokin's yes first): push `public` → GitHub Pages rebuilds; then on the
  box `sudo svc-update` pulls; verify `curl -sI https://solve-vision-with-code.com/ leaderboard.html paper.pdf` = 200
  and one sampled mp4 = 200, and the Pages URL still 200.

## 8 Integration checklist (parent)

1. `git -C project/solve-vision-with-code status` — only `docs/` and `README.md` (one line) changed.
2. `python3 docs/build_site.py --no-media` → `SELF-CHECK OK` (needs the lane results present; if the run outputs are
   not local, run `python3 docs/verify-site.py` alone and say so).
3. `python3 docs/verify-site.py` → exit 0.
4. `python3 -m http.server 8012 -d docs` → `/` and `/leaderboard.html` at 375 and 1280 px: filter pills work and
   update the count; tiles are square and play when visible; a tile opens the panel after its row with five rows ×
   (first frame · Codex · Claude Code · Gemini CLI · Ground truth); `program` opens the code block; `▸ Open-weight
   examples` expands; the big numbers read 0.923 vs 0.670; the leaderboard sorts; the dumbbell resizes; no horizontal
   page scroll at 375.
5. Also open `/solve-vision-with-code/` under a sub-path (`python3 -m http.server -d ..` from the parent dir and
   browse `/docs/`): every path must still resolve (relative paths, no leading `/`).
6. Commit on `public` with a message that says the redesign was done at Hokin's request; no `Co-Authored-By`
   trailer. Push only on Hokin's yes (public repo + GitHub Pages = publishing).

## 9 Things a seat is likely to get wrong

- Renaming `Claude Code/Fable 5.1` to anything else. Site B's names are the data's; the Site A rule about the
  replaced model does not apply here.
- 16:9 tiles. The videos are square; cropping them hides the task.
- Absolute paths (`/media/…`). The site is also served under `/solve-vision-with-code/`.
- Loading `tasks.json` (≈ several MB) on the leaderboard page. It only needs `leaderboard.json`.
- Keeping the sticky nav, the section numbers (`01 Leaderboard`), the heatmap, or the `hokindeng.com` hero button.
  The shape is hero → rule → grid.
- Ground truth first. It is the last column, in accent.
- Dropping the two big numbers or the cite block; both stay.
- A `Co-Authored-By` trailer, or a commit message that does not say the redesign was done at Hokin's request.
- Editing anything under `data/`, `media/`, or `paper.pdf`, or "fixing" a number.
