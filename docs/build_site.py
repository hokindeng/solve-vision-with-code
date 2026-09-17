#!/usr/bin/env python3
"""Build the data and media that the static site in docs/ serves.

    python3 docs/build_site.py [--media-index PATH] [--force] [--jobs N] [--no-media] [--open-per-task N]

Contract: paper/figures/STYLE.md §11 (data) and §12 (size budget). This script is the only writer of
docs/data, docs/media, docs/paper.pdf and docs/.nojekyll; index.html / style.css / app.js are hand-written.

Inputs (read-only)
  <media-index>            index.json written by the media export (default ~/Workspace/svc-media/index.json)
  bench/paper/table*.csv   paper tables (paper_tables.py)
  bench/results/<lane>/<lane>_vbvr_results.json   official evaluator output — the source of truth for every score
  paper/main.pdf           the paper

Outputs (all under docs/; everything goes straight into git — never LFS, GitHub Pages serves pointers)
  data/tasks.json          {generated, lanes, closed_lanes, open_lanes, tasks: [100 × task]}
  data/leaderboard.json    {rows: table1, categories: table2, efficiency: table3}  (CSV header = JSON key)
  data/solve/<lane>/<short>/<idx>.py              solve.py written by the closed-model agents (1 500)
  media/gt/<split>/<short>/<idx>.mp4 | <idx>_first.jpg | <idx>_last.jpg
  media/<lane>/<split>/<short>/<idx>.mp4 | <idx>.jpg   closed lanes: all 500; open lanes: the open_examples only
  paper.pdf  .nojekyll

Every path in the JSON is relative to docs/ with no leading slash; <split> is the verbatim official folder
(In-Domain_50 / Out-of-Domain_50); <short> is the task id minus "_data-generator".
Every mp4 is re-encoded (libx264 main, yuv420p, crf 26, faststart, no audio, native 1024²). Every frame is a
384×384 JPEG q82. open_examples = per task, per open lane, the N (default 2) highest-scoring produced instances
(ties → lower idx).

Self-check (fails loudly): 100 tasks, 500 instances, 1 500 solve files; every score equals the official
evaluator's to 1e-9; every task mean equals table4 to 1e-9; every referenced path exists; leaderboard has
37 rows whose top overall is the CSV maximum (0.923); paper.pdf present; size within the §12 budget.
"""
import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
MEDIA = DOCS / "media"

LANES = {
    "codex": {"label": "Codex / gpt-6-astra", "kind": "agent", "run": "codex-gpt-6-astra"},
    "claude": {"label": "Claude Code / Fable 5.1", "kind": "agent", "run": "claude-fable-5.1-bedrock"},
    "gemini": {"label": "Gemini CLI / 3.1 Pro", "kind": "agent", "run": "gemini-gemini-3.1-pro-preview"},
    "glm-5": {"label": "GLM-5 (OpenCode)", "kind": "agent-open", "run": "bedrock-glm-5"},
    "kimi-k2.5": {"label": "Kimi K2.5 (OpenCode)", "kind": "agent-open", "run": "bedrock-kimi-k2.5"},
    "deepseek-v3.2": {"label": "DeepSeek V3.2 (OpenCode)", "kind": "agent-open", "run": "bedrock-deepseek-v3.2"},
}
CLOSED = ["codex", "claude", "gemini"]
OPEN = ["glm-5", "kimi-k2.5", "deepseek-v3.2"]
INDEX_KEY = {l: l for l in CLOSED} | {l: f"open-{l}" for l in OPEN}   # lane → key inside index.json instances[].lanes
TABLE4_COL = {"codex": "Codex/gpt-6-astra", "claude": "Claude Code/Fable 5.1", "gemini": "Gemini CLI/3.1 Pro"}
THUMB_PX, JPEG_Q = 384, 82
# STYLE.md §11 recipe, plus an explicit limited-range conversion so yuvj420p / gbrp / yuv444p sources come out as plain yuv420p.
FFMPEG = ["-vf", "scale=in_range=auto:out_range=tv,format=yuv420p", "-color_range", "tv",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main", "-crf", "26", "-movflags", "+faststart", "-an"]
BUDGET_MB, HARD_STOP_MB = 300, 500


def die(msg):
    print(f"SELF-CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def num(s):
    f = float(s)
    return int(f) if f.is_integer() and "." not in s else f


# ---------------------------------------------------------------- media jobs (each: src, dst, force → tag)
def video_job(src, dst, force):
    if dst.exists() and not force:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "2", "-i", str(src), *FFMPEG, str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"ffmpeg failed on {src}: {r.stderr[-400:]}")
    os.replace(tmp, dst)
    return "video"


def thumb_job(src, dst, force):
    from PIL import Image
    if dst.exists() and not force:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    if im.width == im.height:
        im = im.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
    else:
        im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
    im.save(dst, "JPEG", quality=JPEG_Q, optimize=True)
    return "thumb"


def solve_job(src, dst, force):
    if dst.exists() and not force:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(Path(src).read_text(errors="replace"))
    return "solve"


# ---------------------------------------------------------------- data
def build_leaderboard():
    def rows(name):
        return [{k: (v if k in ("model", "kind") else num(v)) for k, v in r.items()} for r in csv.DictReader(open(ROOT / "bench/paper" / name))]
    lb = {"rows": rows("table1_leaderboard.csv"), "categories": rows("table2_categories.csv"), "efficiency": rows("table3_efficiency.csv")}
    (DATA / "leaderboard.json").write_text(json.dumps(lb, indent=1))
    return lb


def pick_open_examples(t, per_task):
    """Per open lane: the `per_task` highest-scoring produced instances of this task (ties → lower idx)."""
    out = []
    for lane in OPEN:
        cands = [(-inst["lanes"][INDEX_KEY[lane]]["score"], inst["idx"]) for inst in t["instances"]
                 if inst["lanes"].get(INDEX_KEY[lane], {}).get("video")]
        out += [(lane, idx) for _, idx in sorted(cands)[:per_task]]
    return out


def build_tasks(index, args):
    jobs, tasks_out = [], []
    for tid in sorted(index["tasks"]):
        t = index["tasks"][tid]
        split, short = t["split"], t["short"]
        gt_dir, insts = f"media/gt/{split}/{short}", []
        for inst in t["instances"]:
            idx = inst["idx"]
            gt = {"first_frame": f"{gt_dir}/{idx}_first.jpg", "last_frame": f"{gt_dir}/{idx}_last.jpg", "video": f"{gt_dir}/{idx}.mp4"}
            jobs += [(thumb_job, inst["gt"]["first_frame"], gt["first_frame"]), (thumb_job, inst["gt"]["final_frame"], gt["last_frame"]),
                     (video_job, inst["gt"]["video"], gt["video"])]
            lanes = {}
            for lane in CLOSED:
                e = inst["lanes"][lane]
                d = f"media/{lane}/{split}/{short}"
                rel = {"score": e["score"], "video": None, "last_frame": None, "solve_url": f"data/solve/{lane}/{short}/{idx}.py"}
                if e.get("video"):
                    rel.update(video=f"{d}/{idx}.mp4", last_frame=f"{d}/{idx}.jpg")
                    jobs += [(video_job, e["video"], rel["video"]), (thumb_job, e["last_frame"], rel["last_frame"])]
                jobs.append((solve_job, e["solve"], rel["solve_url"]))
                lanes[lane] = rel
            insts.append({"idx": idx, "prompt": inst["prompt"], "gt": gt, "lanes": lanes})
        opens = []
        by_idx = {i["idx"]: i for i in t["instances"]}
        for lane, idx in pick_open_examples(t, args.open_per_task):
            e, d = by_idx[idx]["lanes"][INDEX_KEY[lane]], f"media/{lane}/{split}/{short}"
            rel = {"lane": lane, "idx": idx, "score": e["score"], "video": f"{d}/{idx}.mp4", "last_frame": f"{d}/{idx}.jpg"}
            jobs += [(video_job, e["video"], rel["video"]), (thumb_job, e["last_frame"], rel["last_frame"])]
            opens.append(rel)
        tasks_out.append({"id": tid, "short": short, "split": split, "category": t["category"],
                          "prompt": insts[0]["prompt"],
                          "mean": {lane: sum(i["lanes"][lane]["score"] for i in insts) / len(insts) for lane in CLOSED},
                          "gt": insts[0]["gt"], "instances": insts, "open_examples": opens})

    counts = {}
    if not args.no_media:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for i, tag in enumerate(ex.map(lambda j: j[0](j[1], DOCS / j[2], args.force), jobs), 1):
                counts[tag] = counts.get(tag, 0) + 1
                if i % 1000 == 0:
                    print(f"  media {i}/{len(jobs)} {counts}", flush=True)
    doc = {"generated": dt.date.today().isoformat(), "lanes": LANES, "closed_lanes": CLOSED, "open_lanes": OPEN, "tasks": tasks_out}
    (DATA / "tasks.json").write_text(json.dumps(doc, indent=1))
    return doc, counts


# ---------------------------------------------------------------- self-check
def official_scores(run):
    j = json.load(open(ROOT / "bench/results" / run / f"{run}_vbvr_results.json"))
    return {(s["task_name"], Path(s["video_file"]).stem): s["score"] for s in j["samples"]}


def self_check(doc, lb):
    tasks = doc["tasks"]
    if len(tasks) != 100:
        die(f"tasks = {len(tasks)}, expected 100")
    n_inst = sum(len(t["instances"]) for t in tasks)
    if n_inst != 500:
        die(f"instances = {n_inst}, expected 500")
    official = {lane: official_scores(meta["run"]) for lane, meta in LANES.items()}
    table4 = {r["task"]: r for r in csv.DictReader(open(ROOT / "bench/paper/table4_per_task_closed.csv"))}
    missing, n_paths, n_scores = [], 0, 0

    def check_path(p):
        nonlocal n_paths
        if p is None:
            return
        n_paths += 1
        if p.startswith("/") or not (DOCS / p).is_file():
            missing.append(p)

    def check_score(lane, tid, idx, score):
        nonlocal n_scores
        ref = official[lane].get((tid, idx))
        if ref is None or abs(ref - score) > 1e-9:
            die(f"score mismatch {lane} {tid} {idx}: site {score} official {ref}")
        n_scores += 1

    for t in tasks:
        if t["split"] not in ("In-Domain_50", "Out-of-Domain_50"):
            die(f"bad split {t['split']}")
        if [i["idx"] for i in t["instances"]] != [f"{k:05d}" for k in range(5)]:
            die(f"{t['id']}: instances are not 00000..00004")
        for p in t["gt"].values():
            check_path(p)
        for inst in t["instances"]:
            for p in inst["gt"].values():
                check_path(p)
            for lane, e in inst["lanes"].items():
                for k in ("video", "last_frame", "solve_url"):
                    check_path(e[k])
                check_score(lane, t["id"], inst["idx"], e["score"])
        row = table4.get(t["short"]) or die(f"{t['short']} missing from table4")
        for lane in CLOSED:
            if abs(t["mean"][lane] - float(row[TABLE4_COL[lane]])) > 1e-9:
                die(f"mean mismatch {lane} {t['short']}: {t['mean'][lane]} vs table4 {row[TABLE4_COL[lane]]}")
        for o in t["open_examples"]:
            check_path(o["video"]); check_path(o["last_frame"])
            check_score(o["lane"], t["id"], o["idx"], o["score"])
    if missing:
        die(f"{len(missing)} referenced paths missing, e.g. {missing[:5]}")
    n_solve = sum(1 for _ in (DATA / "solve").rglob("*.py"))
    if n_solve != 1500:
        die(f"solve files on disk = {n_solve}, expected 1500")

    rows = lb["rows"]
    csv_best = max(float(r["overall"]) for r in csv.DictReader(open(ROOT / "bench/paper/table1_leaderboard.csv")))
    if len(rows) != 37:
        die(f"leaderboard rows = {len(rows)}, expected 37")
    if rows[0]["overall"] != csv_best or round(rows[0]["overall"], 3) != 0.923:
        die(f"leaderboard top row {rows[0]} != CSV best {csv_best}")
    if not (DOCS / "paper.pdf").is_file():
        die("docs/paper.pdf missing")

    size = lambda p: sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6
    media_mb, data_mb = size(MEDIA), size(DATA)
    big = [str(f) for f in DOCS.rglob("*") if f.is_file() and f.stat().st_size > 100e6]
    if big:
        die(f"files over 100 MB: {big}")
    if media_mb + data_mb > HARD_STOP_MB:
        die(f"docs/ media+data = {media_mb + data_mb:.0f} MB > hard stop {HARD_STOP_MB} MB")
    if media_mb + data_mb > BUDGET_MB:
        print(f"WARNING: docs/ media+data = {media_mb + data_mb:.0f} MB exceeds the {BUDGET_MB} MB target", file=sys.stderr)
    return {"tasks": len(tasks), "instances": n_inst, "scores_verified": n_scores, "paths_verified": n_paths, "solve_files": n_solve,
            "open_examples": sum(len(t["open_examples"]) for t in tasks), "mp4": sum(1 for _ in MEDIA.rglob("*.mp4")),
            "jpg": sum(1 for _ in MEDIA.rglob("*.jpg")), "leaderboard_rows": len(rows),
            "top": f"{rows[0]['model']} {rows[0]['overall']:.3f}", "media_mb": round(media_mb, 1), "data_mb": round(data_mb, 1),
            "paper_pdf_kb": round((DOCS / "paper.pdf").stat().st_size / 1e3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media-index", default=str(Path.home() / "Workspace/svc-media/index.json"))
    ap.add_argument("--force", action="store_true", help="rebuild media that already exists")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--no-media", action="store_true", help="only rewrite the JSON (media must already exist for the self-check)")
    ap.add_argument("--open-per-task", type=int, default=2, help="open-lane instances shipped per task per lane (STYLE.md §12: at most 2)")
    args = ap.parse_args()

    index = json.load(open(args.media_index))
    DATA.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").touch()

    lb = build_leaderboard()
    print(f"leaderboard.json: rows {len(lb['rows'])}, categories {len(lb['categories'])}, efficiency {len(lb['efficiency'])}")
    doc, counts = build_tasks(index, args)
    print(f"media jobs: {counts}")
    pdf = ROOT / "paper/main.pdf"
    if not pdf.is_file():
        die("paper/main.pdf not found")
    shutil.copyfile(pdf, DOCS / "paper.pdf")

    print("SELF-CHECK OK", json.dumps(self_check(doc, lb), indent=1))


if __name__ == "__main__":
    main()
