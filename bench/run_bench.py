#!/usr/bin/env python3
"""Run a coding agent on VBVR-Pro-Bench (the official 100-task benchmark) in the TI2V setting.

    OPENAI_API_KEY=... python3 bench/run_bench.py --model gpt-6-astra [--tasks G-18,...|all]
        [--instances 0,1,2,3,4] [--split both] [--parallel 3] [--no-build]

Inputs per instance (exactly what video models get): first_frame.png + prompt.txt from
project/vbvr-pro-bench-data/VBVR-Pro-Bench-Video/<split>/<task>/<idx>/. The agent additionally gets the
target container spec (size/fps/frames probed from the GT — disclosed condition, identical across models
we run). Output: bench/outputs/<model>/<split>/<task>/<idx>.mp4 — the official evaluator's expected
layout — plus the full agent workspace under bench/runs/<model>/<split>/<task>/<idx>/.
Score afterwards with VBVR-Pro-Bench/run_evaluation_video.py --model_path bench/outputs/<model> --gt_base <data>.
"""
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "vbvr-pro-bench-data" / "VBVR-Pro-Bench-Video"
IMAGE = "svc-agent"
INSTRUCTION = """# {task} · instance {idx}

`/app/first_frame.png` is the first frame of a short video. `/app/prompt.txt` describes what happens
in that video. Write code that produces the video.

Prompt:

> {prompt}

## Deliverables

Write everything under `/app`. Required:

- `/app/output/video.mp4` — H.264, yuv420p, {w}x{h}, {fps} fps, about {frames} frames ({dur} s). Its
  first frame is `first_frame.png`; its last frame shows the completed result.
- Keep the program at `/app/solve.py` so `python3 /app/solve.py` regenerates the video.

## Rules

- The video is drawn by your code (numpy, Pillow, OpenCV, imageio, ffmpeg are installed; you may install
  more). Do not call an image or video generation model.
- Change only what the prompt asks for; every other pixel stays as in `first_frame.png`, in every frame.
- Pace the action over the full duration rather than jumping to the end.
"""


def probe(video: Path):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, int(fps) if float(fps).is_integer() else round(fps, 2), n


def run_instance(name, agent, model, reasoning, split, task, idx, timeout):
    inst = DATA / split / task / idx
    run = ROOT / "bench" / "runs" / name / split / task / idx
    out_mp4 = ROOT / "bench" / "outputs" / name / split / task / f"{idx}.mp4"
    if out_mp4.exists():
        return dict(task=task, idx=idx, skipped=True)
    if os.environ.get("SVC_SKIP_ATTEMPTED") and (run / "summary.json").exists():  # resume: one attempt per instance
        return dict(task=task, idx=idx, skipped=True)
    app = run / "app"
    if run.exists():
        shutil.rmtree(run)
    app.mkdir(parents=True)
    shutil.copy(inst / "first_frame.png", app / "first_frame.png")
    shutil.copy(inst / "prompt.txt", app / "prompt.txt")
    w, h, fps, n = probe(inst / "ground_truth.mp4")
    prompt = (inst / "prompt.txt").read_text().strip().replace("\n", "\n> ")
    (app / "instruction.md").write_text(INSTRUCTION.format(task=task, idx=idx, prompt=prompt, w=w, h=h, fps=fps, frames=n, dur=round(n / float(fps), 2)))
    os.chmod(app, 0o777)
    t0 = time.time()
    passthrough = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                   "AWS_SESSION_TOKEN", "AWS_REGION", "CLAUDE_CODE_USE_BEDROCK", "SVC_BASE_URL", "SVC_API_KEY", "SVC_PROVIDER", "SVC_BEDROCK_REGION", "SVC_MAX_OUTPUT"]
    env = [x for k in passthrough if os.environ.get(k) for x in ("-e", k)]
    env += ["-e", f"SVC_AGENT={agent}", "-e", f"SVC_MODEL={model}"] + (["-e", f"SVC_REASONING={reasoning}"] if reasoning else [])
    limits = ["--memory", os.environ.get("SVC_MEM", "3g"), "--memory-swap", os.environ.get("SVC_MEM", "3g"),
              "--cpus", os.environ.get("SVC_CPUS", "2"), "--pids-limit", "512"]  # a runaway solve.py dies in its own box
    name = f"svcb-{task[:20]}-{idx}-{os.getpid()}"
    try:
        r = subprocess.run(["docker", "run", "--rm", "--name", name, *limits, *env, "-v", f"{app}:/app", IMAGE],
                           capture_output=True, text=True, timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:  # the agent overran: kill its container, count the instance as failed, keep the lane going
        subprocess.run(["docker", "kill", name], capture_output=True)
        rc = -9
    ok = (app / "output" / "video.mp4").exists()
    if ok:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(app / "output" / "video.mp4", out_mp4)
    res = dict(task=task, idx=idx, agent_exit=rc, seconds=round(time.time() - t0), video=ok)
    (run / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--agent", default="codex", choices=["codex", "claude", "gemini", "opencode"])
    ap.add_argument("--name", default="", help="output dir name; default <agent>-<model>")
    ap.add_argument("--reasoning", default="")
    ap.add_argument("--tasks", default="all", help="comma list of task-dir prefixes, or 'all'")
    ap.add_argument("--instances", default="0,1,2,3,4")
    ap.add_argument("--split", default="both", choices=["both", "In-Domain_50", "Out-of-Domain_50"])
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    needed = {"codex": ["OPENAI_API_KEY"], "claude": ["ANTHROPIC_API_KEY", "AWS_SESSION_TOKEN", "AWS_ACCESS_KEY_ID"],
              "gemini": ["GEMINI_API_KEY"], "opencode": ["SVC_API_KEY"]}[a.agent]
    if not any(os.environ.get(k) for k in needed):
        sys.exit(f"none of {needed} set")
    name = a.name or f"{a.agent}-{a.model}"
    if not a.no_build:
        subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(ROOT / "bench" / "agent")], check=True)
    splits = ["In-Domain_50", "Out-of-Domain_50"] if a.split == "both" else [a.split]
    idxs = [f"{int(i):05d}" for i in a.instances.split(",")]
    jobs = []
    for split in splits:
        for tdir in sorted(p.name for p in (DATA / split).iterdir() if p.is_dir()):
            if a.tasks != "all" and not any(tdir.startswith(t.strip()) for t in a.tasks.split(",")):
                continue
            for idx in idxs:
                if (DATA / split / tdir / idx).exists():
                    jobs.append((split, tdir, idx))
    print(f"{len(jobs)} instances", flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        results = list(ex.map(lambda j: run_instance(name, a.agent, a.model, a.reasoning, *j, a.timeout), jobs))
    done = [r for r in results if r.get("video")]
    print(json.dumps(dict(total=len(jobs), produced=len(done), skipped=sum(1 for r in results if r.get("skipped")))))


if __name__ == "__main__":
    main()
