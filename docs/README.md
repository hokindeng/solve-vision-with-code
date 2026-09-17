# docs/ — the results site

Static site for *Solve Vision with Code*, served by GitHub Pages at
https://hokindeng.github.io/solve-vision-with-code/. It shows the full VBVR-Pro-Bench leaderboard, the
category and efficiency tables, the per-task heatmap, and a gallery with every agent video and `solve.py`
of the three closed-model lanes (Codex × gpt-6-astra, Claude Code × Fable 5.1, Gemini CLI × 3.1 Pro;
1,500 instances) next to the ground-truth videos, plus a few open-weight examples. `paper.pdf` is the paper.

```
index.html, style.css, app.js   the page; app.js reads data/ and media/
build_site.py                   regenerates data/ and media/ from the run outputs
data/                           leaderboard, tables, per-task scores (JSON); data/solve/ = the 1,500 programs
media/                          agent videos, ground-truth videos, thumbnails
paper.pdf                       the paper
```

## Rebuild

From the repository root, with `ffmpeg` on the path and the lane output directories present (the
`bench/outputs/<lane>/` videos and `bench/runs/<lane>/` workspaces of the three closed lanes, plus the
official benchmark data for the ground-truth videos; a local copy or an S3 sync of the bucket in
`../DATA_MANIFEST.md`):

```sh
python3 docs/build_site.py
```

The script reads `bench/results/`, `bench/stats.json` and `bench/paper/` for the numbers, copies the videos
and programs into `media/` and `data/solve/`, and writes the thumbnails. Everything under `data/` and
`media/` is generated; edit the sources, not the output.

## Preview

```sh
cd docs && python3 -m http.server 8123
```

Then open http://localhost:8123/.

## Pages

GitHub Pages serves branch `public`, path `/docs`. A push to `public` redeploys; no build step runs on
GitHub.
