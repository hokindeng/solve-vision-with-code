# solve-vision-with-code

Code and data for *Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models*.
A coding agent is given the first frame and the prompt of a VBVR-Pro-Bench instance and writes a program
that renders the answer video; the video is scored by the benchmark's official rule-based evaluator, so
its score sits on the same leaderboard as the video generation models. This repository holds the agent
harness (`bench/`), the per-lane evaluator results and run statistics behind every number in the paper,
and the paper source (`paper/`). Project page: https://hokindeng.com.

## Results

VBVR-Pro-Bench, video setting: 100 tasks (50 In-Domain, 50 Out-of-Domain) × 5 instances = 500 per lane.
Scores are the unmodified official evaluator (`task_specific_only`, CPU), mean over all 500 instances;
an instance without a video counts 0. Video-model rows are the published VBVR-Pro-Bench numbers. Full table
with every video model: `bench/paper/tables.md`; source of truth: `bench/paper/table1_leaderboard.csv`.

| # | Model | Agent | Overall | In-Domain | Out-of-Domain |
|---|---|---|---|---|---|
| 1 | gpt-6-astra | Codex CLI 0.154.0 | **0.923** | 0.879 | 0.967 |
| 2 | Claude Fable 5.1 (Amazon Bedrock) | Claude Code 2.1.268 | **0.894** | 0.830 | 0.958 |
| 3 | gemini-3.1-pro-preview | Gemini CLI 0.59.0 | **0.893** | 0.826 | 0.960 |
| 4 | VBVR-Pro-Wan2.2-I2V-A14B (best video model; RL-trained on the In-Domain families) | — | 0.670 | 0.808 | 0.532 |
| 5 | GLM-5 | OpenCode 1.18.30 | 0.572 | 0.479 | 0.664 |
| 7 | Kimi K2.5 | OpenCode | 0.557 | 0.454 | 0.660 |
| 11 | MiniMax M2.5 | OpenCode | 0.451 | 0.386 | 0.515 |
| 14 | DeepSeek V3.2 | OpenCode | 0.378 | 0.303 | 0.452 |
| 15 | Devstral 2 123B | OpenCode | 0.350 | 0.275 | 0.426 |
| 16 | Kimi K2 Thinking | OpenCode | 0.313 | 0.218 | 0.407 |
| 18 | gpt-oss-120b | OpenCode | 0.270 | 0.196 | 0.344 |
| 19 | Qwen3-Coder-Next | OpenCode | 0.264 | 0.218 | 0.310 |
| 20 | Nemotron Super 3 120B | OpenCode | 0.200 | 0.139 | 0.261 |
| 22 | Qwen3-Coder-30B-A3B | OpenCode | 0.143 | 0.099 | 0.186 |
| 23 | Mistral Large 3 675B | OpenCode | 0.127 | 0.079 | 0.176 |
| 25 | GLM-4.7-Flash | OpenCode | 0.104 | 0.081 | 0.127 |
| 27 | gpt-oss-20b | OpenCode | 0.079 | 0.075 | 0.083 |
| 28 | Qwen3-Next-80B-A3B | OpenCode | 0.070 | 0.034 | 0.105 |
| 29 | Llama 4 Scout | OpenCode | 0.049 | 0.040 | 0.058 |
| 30 | Nemotron Nano 3 30B | OpenCode | 0.028 | 0.026 | 0.029 |
| 31 | Qwen3-VL-235B-A22B | OpenCode | 0.025 | 0.022 | 0.028 |
| 32 | Qwen3-32B | OpenCode | 0.004 | 0.006 | 0.003 |
| 33 | Llama 4 Maverick | OpenCode | 0.001 | 0.001 | 0.002 |
| 34–37 | Gemma 3 27B, Gemma 3 12B, Magistral Small, Llama 3.3 70B (no video in 500 attempts) | OpenCode | 0.000 | 0.000 | 0.000 |

Ranks are positions in the full leaderboard (video models interleaved). All 24 open-weight models were
served by Amazon Bedrock; DeepSeek R1 is not on the table because Bedrock offers no tool use for it.
Every coding agent above the noise floor, closed or open, scores higher Out-of-Domain than In-Domain, with no
training on any family; every video model trained on the In-Domain families scores lower Out-of-Domain, while
the untrained video models move by less than 0.1.

## Layout

```
bench/
  run_bench.py              run one agent on the 500 instances inside Docker; writes outputs/ and runs/
  run_roster.sh             the 24 open-weight lanes, one after another, through OpenCode
  roster_bedrock_open.json  the open-weight roster: Bedrock model id + route per lane
  bedrock_proxy.py          OpenAI-compatible shim over Bedrock's non-streaming Converse API (Llama lanes)
  report.py                 leaderboard table + HTML page from results/
  stats.py                  per-lane run statistics from the agents' event logs -> stats.json
  paper_tables.py           the paper's tables (CSV + markdown) and figures from results/ + stats.json
  results/<lane>/           official evaluator output per lane (<lane>_vbvr_results.json)
  paper/                    table1_leaderboard.csv, table2_categories.csv, table3_efficiency.csv,
                            table4_per_task_closed.csv, tables.md
  agent/                    the agent image: python 3.11, ffmpeg, numpy, Pillow, OpenCV, imageio, scipy,
                            matplotlib, and the four agent CLIs at pinned versions; entrypoint.sh
paper/                      the paper (ICML template); `paper/build.sh` builds main.pdf
```

## Reproduce one lane

**1. Data.** Download VBVR-Pro-Bench from Hugging Face (`Video-Reason/VBVR-Pro-Bench`) and unpack it next
to this repository, so that this path exists:

```
../vbvr-pro-bench-data/VBVR-Pro-Bench-Video/{In-Domain_50,Out-of-Domain_50}/<task>/<idx>/{first_frame.png,prompt.txt,ground_truth.mp4}
```

`bench/run_bench.py` reads `DATA` from that location. The agent is shown only `first_frame.png` and
`prompt.txt`; `ground_truth.mp4` is probed for the container spec (size, fps, frame count) that is written
into `instruction.md`, and is never mounted into the container.

**2. Agent image.** `run_bench.py` builds `bench/agent/` as `svc-agent` on first use (`--no-build` skips
it). One container per instance: 2 CPUs, 3 GB RAM, 512 pids, 1800 s wall clock, one attempt, only `/app`
mounted. The agent must leave `/app/output/video.mp4` and `/app/solve.py`; it may `pip install`, it may not
call an image or video generation model.

**3. Run.** Outputs land in `bench/outputs/<lane>/<split>/<task>/<idx>.mp4` (the evaluator's layout) and the
full workspace of every instance (solve.py, `events.jsonl`, `output/`, `summary.json`) in
`bench/runs/<lane>/<split>/<task>/<idx>/`.

```sh
# Codex CLI x gpt-6-astra
OPENAI_API_KEY=... python3 bench/run_bench.py --agent codex --model gpt-6-astra --parallel 3 --timeout 1800

# Claude Code x Claude Fable 5.1 through Amazon Bedrock (AWS credentials in the environment)
CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-east-1 AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=... \
  python3 bench/run_bench.py --agent claude --model us.anthropic.claude-fable-5-1 --name claude-fable-5.1-bedrock

# Gemini CLI x gemini-3.1-pro-preview
GEMINI_API_KEY=... python3 bench/run_bench.py --agent gemini --model gemini-3.1-pro-preview

# OpenCode x any OpenAI-compatible endpoint (this is how the open-weight lanes ran)
SVC_API_KEY=... SVC_BASE_URL=https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1 \
  python3 bench/run_bench.py --agent opencode --model zai.glm-5 --name bedrock-glm-5 --parallel 12
```

`--tasks G-18,O-1` and `--instances 0,1` select a subset; `SVC_SKIP_ATTEMPTED=1` resumes a lane without
giving an already-attempted instance a second try; `SVC_MEM`, `SVC_CPUS` change the container limits.

**4. Score** with the official kit ([Video-Reason/VBVR-Pro-Bench](https://github.com/Video-Reason/VBVR-Pro-Bench)),
unmodified, from a virtualenv built from the kit's own `requirements.txt`:

```sh
python -m venv ~/venv-eval && ~/venv-eval/bin/pip install -r VBVR-Pro-Bench/requirements.txt "numpy<2"
~/venv-eval/bin/python VBVR-Pro-Bench/run_evaluation_video.py --device cpu \
  --model_path bench/outputs/<lane> --gt_base ../vbvr-pro-bench-data/VBVR-Pro-Bench-Video \
  --output_dir bench/results/<lane>
```

> **Evaluator environment — read before trusting a score.** The kit needs `numpy<2` (norfair requires it),
> `norfair` and `easyocr`. In an environment that violates this, four evaluators (G-5, G-8, O-18, O-19)
> silently return 0 for every video, ground truth included, with no error in the log. Keep the scoring venv
> separate from any numpy-2 / OpenCV-5 generator venv, and before scoring a lane score the kit's own
> ground-truth videos: every one must come back 1.0. A task that is exactly 0.0 for every model is an
> evaluator problem until proven otherwise.

**5. Tables.**

```sh
python3 bench/report.py bench/leaderboard.html            # leaderboard table (markdown + HTML) from bench/results/
python3 bench/stats.py bench/runs bench/stats.json        # tool calls, tokens, wall time, produced/timeouts per lane
python3 bench/paper_tables.py bench/paper                 # table1..4 CSV + tables.md + figures
```

`report.py` carries the lane-name map and the published video-model rows; add a new lane there.

## The 24 open-weight lanes

`bench/run_roster.sh [parallel] [lane ...]` runs every cell of `bench/roster_bedrock_open.json` through
OpenCode with `SVC_API_KEY` (a Bedrock bearer token). Three routes:

- `bedrock` (20 lanes): Bedrock's OpenAI-compatible endpoint, `SVC_PROVIDER=openai`.
- `bedrock-proxy` (Llama 4 Scout, Llama 4 Maverick, Llama 3.3 70B): the OpenAI-compatible endpoint returns
  404 for these ids and their streaming Converse API rejects tool use, so `bench/bedrock_proxy.py` serves an
  OpenAI-compatible chat endpoint on the Docker host that calls non-streaming Converse and replays the reply as
  a short SSE stream. Start it with `AWS_BEARER_TOKEN_BEDROCK=... python3 bench/bedrock_proxy.py --port 8765`;
  containers reach it at `SVC_PROXY_URL` (default `http://172.17.0.1:8765/v1`, the Docker bridge).
- `unsupported` (DeepSeek R1): no tool use on Bedrock in either mode; skipped.

OpenCode's max output is capped at 8192 tokens on the native Bedrock route (`SVC_MAX_OUTPUT` overrides).
The agents ran 2026-09-12 to 2026-09-16 on one c7i.4xlarge (CPU only).

## Known limitations of the source tasks

For the reward check (paper, appendix) all 300 VBVR-Pro generators were packaged as tasks, prompt and
truth taken from the generator as they are; that task set is not part of this release (the generators are
not ours to publish). Reading every prompt against its first frame surfaced recurring gaps, listed here by
task id (T numbering follows the `generator-fixes` set of the source; G/O ids are the benchmark's):

- **Truth or prompt is wrong.** T094, T097 (fixed by a seed override), T118 (prompt says cubes stay blue;
  truth ends grey), O-9 (a "200 %" enlargement renders as a 0.86 shrink), O-44 (92°/105° under a "90°" rule),
  O-52, O-62, O-7, O-11/O-12 (colour names print as `unknown` / `color_NNN`). O-13 is patched to match its
  render.
- **Answer not unique.** G-4/T004, G-13, G-16, G-41, T012 (tied shortest routes); T080, T083 (several legal
  moves); T113; T135 (42 valid nonograms); T141; T142.
- **Prompt written for a text or multiple-choice answer.** T054, T061, T084, T086, T090, T096, T100, T104,
  T108, T114, T115, T120, T121, O-55.
- **Essential state only in metadata.** T031, T033, T007, T028, T049/T050, T077, T102, T116, T119, G-3,
  G-54, O-16/O-17, O-66, O-87.
- **Truth adds elements the prompt never mentions** (captions, highlight boxes, construction lines):
  T032, T056, T058, T059, T088, T089, T099, T101, T105, T117, T120, T122, T123, T124, T144.
- **Styling is unstated** (ring radius, stroke width, exact shade, pacing) on most G-series marking tasks.
- **Rendering.** Truth videos were rendered on macOS; generators that look up system fonts by name can
  render differently inside a Linux image. T036 and T037 need Blender.

## Data

Everything the paper's numbers rest on is in the repository: `bench/results/<lane>/` (the evaluator's
per-instance JSON for the 22 lanes that produced a video), `bench/stats.json`, `bench/paper/`. The agent
videos of those 22 lanes and the per-instance agent workspaces (solve.py, event logs) of all 27 lanes are
held privately (layout in `DATA_MANIFEST.md`); contact the corresponding author for access.

## Citation

```bibtex
@article{deng2026solvevision,
  title   = {Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models},
  author  = {Deng, Hokin and others},
  year    = {2026},
  note    = {https://github.com/hokindeng/solve-vision-with-code}
}
```

The benchmark, evaluator and video-model numbers are from VBVR-Pro (arXiv 2608.26105).

## License

See `LICENSE`. Research and non-commercial use; contact the copyright holder for anything else.
