# DATA_MANIFEST — solve-vision-with-code

Bucket: `s3://solve-vision-with-code-release/` (us-east-1, private, versioned, AES256). Nothing large lives
on a laptop; the bucket is the record. Object counts and sizes below are from a recursive listing on
2026-09-16. Lane names are the directory names used throughout `bench/` (`codex-gpt-6-astra`,
`claude-fable-5.1-bedrock`, `gemini-gemini-3.1-pro-preview`, `bedrock-<model>` for the open-weight lanes).

| Prefix | Content | Objects | Size |
|---|---|---|---|
| `bench/outputs/<lane>/<split>/<task>/<idx>.mp4` | agent videos in the official evaluator layout, the three closed-model lanes as directories (500 mp4 each) | 1,500 | 75 MB |
| `bench/outputs/<lane>.tgz` | the same layout, one tarball per open-weight lane that produced at least one video (19 lanes: deepseek-v3.2, devstral-2, glm-4.7-flash, glm-5, gpt-oss-120b, gpt-oss-20b, kimi-k2-thinking, kimi-k2.5, llama4-maverick, llama4-scout, minimax-m2.5, mistral-large-3, nemotron-nano-3, nemotron-super-3, qwen3-32b, qwen3-coder-30b, qwen3-coder-next, qwen3-next-80b, qwen3-vl-235b) | 19 | 543 MB |
| `bench/runs/<lane>/<split>/<task>/<idx>/` | per-instance agent workspaces (`app/solve.py`, `app/events.jsonl`, `app/output/`, `summary.json`) for the three closed-model lanes | 25,055 | 415 MB |
| `bench/runs/<lane>.tgz` | the same, one tarball per open-weight lane, all 24 roster cells (the 19 above plus the four that produced no video — gemma-3-27b, gemma-3-12b, magistral-small, llama3.3-70b — and the aborted deepseek-r1 attempt, 401 instances, no tool use) | 24 | 1.15 GB |
| `bench/runs/claude-claude-fable-5-1/` | an aborted first Claude lane (9 objects, one instance); superseded by `claude-fable-5.1-bedrock` | 9 | 12 KB |
| `bench/evaluation_results/<lane>/<lane>_vbvr_results.json` | **the official evaluator output per lane, kit-conforming environment (numpy<2, norfair, easyocr)** — 22 lanes with videos, plus `svcb-oracle/` and `svcb-frozen/` (ground-truth and frozen-first-frame sanity runs); the three closed lanes also carry `all_models_summary.json`. Identical to `bench/results/<lane>/` in the repository, which is where `report.py` reads them. | 29 | 5.9 MB |
| `bench/results/` | **superseded** — the first Codex scoring (2026-09-12, numpy-2 environment; G-5/G-8/O-18/O-19 scored 0 for every video). Same bytes as the copy under `archive/2026-09-13_…`. Do not use. | 2 | 0.3 MB |
| `bench/box-logs-2026-09-16.tgz` | the bench box's roster and scoring logs | 1 | 0.5 MB |
| `official-data/VBVR-Pro-Bench-Video.tar.gz` | the official benchmark data as downloaded from HF (`Video-Reason/VBVR-Pro-Bench`) | 1 | 90 MB |
| `paper/2026-09-16_tables/` | `bench/paper/` as generated for the paper (table1–4 CSV, tables.md, paper_tables.html) | 6 | 0.1 MB |
| `harbor-runs/gpt-6-astra/<task>/` | the first Harbor-task agent runs (2026-09-11, 11 tasks, pre-`main_v2` G/O samples; not used in the paper) | 176 | 0.6 MB |
| `archive/2026-09-11_svc-superseded-tasks-t134-t150/` | the 17 task directories replaced by the final T134–T150 set | 724 | 31 MB |
| `archive/2026-09-13_bench-evaluation-results-numpy2-env-superseded/` | the three closed-model lanes and the oracle/frozen runs as first scored in the non-conforming environment | 10 | 1.2 MB |
| `pages/` | the display pages (run galleries 2026-09-11, bench pages 2026-09-12/13/16) | 41 | 1 MB |

Total: about 27,600 objects, 2.3 GB.

Bench box: one c7i.4xlarge (us-east-1, CPU only) ran every lane 2026-09-12 → 09-16 and wrote into the
same layout; a lane's videos, workspaces and evaluator JSON were copied here when it completed. The box is
stopped.

## Public copies

The bucket stays private. These parts of it are public, as of 2026-09-17:

| Public location | Content | Bucket source |
|---|---|---|
| `bench/results/<lane>/` (GitHub repo) | official evaluator JSON, 22 lanes + oracle/frozen | `bench/evaluation_results/` |
| `bench/stats.json`, `bench/paper/` (GitHub repo) | run statistics; table1–4 CSV, tables.md | `paper/2026-09-16_tables/` |
| `docs/media/` (site, https://hokindeng.github.io/solve-vision-with-code/) | the three closed-model lanes' agent videos (1,500 mp4), the 500 ground-truth videos, and thumbnails | `bench/outputs/<lane>/` (closed lanes), `official-data/` |
| `docs/data/solve/` (site) | the 1,500 `solve.py` programs of the three closed-model lanes | `bench/runs/<lane>/<split>/<task>/<idx>/app/solve.py` |
| `docs/paper.pdf` (site) | the paper | — |

`docs/build_site.py` regenerates `docs/data/` and `docs/media/` from the lane output directories (`docs/README.md`).
Not public: the open-weight lanes' videos (`bench/outputs/<lane>.tgz`), every per-instance workspace beyond
`solve.py` (event logs, container output, `summary.json`), the box logs, and the archive prefixes. These are
available from the corresponding author on request.

## Final leaderboard (2026-09-16, VBVR-Pro-Bench video setting)

Official evaluator, kit-conforming environment, mean over all 500 instances; an instance without a video
counts 0. "Videos" is the number of instances for which the agent left a video (`bench/stats.json`);
video-model rows are the published VBVR-Pro-Bench numbers. Source: `bench/paper/table1_leaderboard.csv`.

| # | Model | Kind | Overall | In-Domain | Out-of-Domain | Videos / 500 |
|---|---|---|---|---|---|---|
| 1 | Codex × gpt-6-astra | coding agent | **0.9228** | 0.8785 | 0.9671 | 500 |
| 2 | Claude Code × Fable 5.1 | coding agent | **0.8943** | 0.8304 | 0.9581 | 500 |
| 3 | Gemini CLI × 3.1 Pro | coding agent | **0.8930** | 0.8263 | 0.9596 | 500 |
| 4 | VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model) | video model | 0.6700 | 0.8080 | 0.5320 | — |
| 5 | OpenCode × GLM-5 | coding agent (open weights) | **0.5719** | 0.4794 | 0.6644 | 495 |
| 6 | VBVR-Pro-Wan2.1-I2V-14B | video model | 0.5620 | 0.7300 | 0.3950 | — |
| 7 | OpenCode × Kimi K2.5 | coding agent (open weights) | **0.5573** | 0.4543 | 0.6603 | 498 |
| 8 | VBVR-Wan2.2 | video model | 0.5170 | 0.5480 | 0.4860 | — |
| 9 | Seedance 2.0 | video model | 0.4990 | 0.4510 | 0.5470 | — |
| 10 | VBVR-Pro-Wan2.2-TI2V-5B | video model | 0.4700 | 0.6410 | 0.3000 | — |
| 11 | OpenCode × MiniMax M2.5 | coding agent (open weights) | **0.4507** | 0.3862 | 0.5153 | 488 |
| 12 | VBVR-Pro-LTX2.3 | video model | 0.4250 | 0.5270 | 0.3240 | — |
| 13 | Kling VIDEO 3.0 | video model | 0.3920 | 0.3560 | 0.4270 | — |
| 14 | OpenCode × DeepSeek V3.2 | coding agent (open weights) | **0.3779** | 0.3034 | 0.4524 | 475 |
| 15 | OpenCode × Devstral 2 123B | coding agent (open weights) | **0.3502** | 0.2748 | 0.4257 | 494 |
| 16 | OpenCode × Kimi K2 Thinking | coding agent (open weights) | **0.3126** | 0.2183 | 0.4070 | 406 |
| 17 | Veo 3.1 | video model | 0.3090 | 0.3120 | 0.3050 | — |
| 18 | OpenCode × gpt-oss-120b | coding agent (open weights) | **0.2700** | 0.1957 | 0.3442 | 380 |
| 19 | OpenCode × Qwen3-Coder-Next | coding agent (open weights) | **0.2639** | 0.2181 | 0.3096 | 498 |
| 20 | OpenCode × Nemotron Super 3 120B | coding agent (open weights) | **0.2003** | 0.1393 | 0.2613 | 436 |
| 21 | Wan2.2-I2V-A14B | video model | 0.1820 | 0.1570 | 0.2070 | — |
| 22 | OpenCode × Qwen3-Coder-30B-A3B | coding agent (open weights) | **0.1427** | 0.0993 | 0.1861 | 480 |
| 23 | OpenCode × Mistral Large 3 675B | coding agent (open weights) | **0.1274** | 0.0786 | 0.1762 | 254 |
| 24 | LTX-2.3-I2AV | video model | 0.1120 | 0.1060 | 0.1190 | — |
| 25 | OpenCode × GLM-4.7-Flash | coding agent (open weights) | **0.1039** | 0.0811 | 0.1268 | 470 |
| 26 | Wan2.1-I2V-14B-720P | video model | 0.1000 | 0.1050 | 0.0950 | — |
| 27 | OpenCode × gpt-oss-20b | coding agent (open weights) | **0.0787** | 0.0745 | 0.0830 | 148 |
| 28 | OpenCode × Qwen3-Next-80B-A3B | coding agent (open weights) | **0.0698** | 0.0341 | 0.1054 | 167 |
| 29 | OpenCode × Llama 4 Scout (Converse proxy) | coding agent (open weights) | **0.0493** | 0.0403 | 0.0583 | 269 |
| 30 | OpenCode × Nemotron Nano 3 30B | coding agent (open weights) | **0.0278** | 0.0261 | 0.0294 | 91 |
| 31 | OpenCode × Qwen3-VL-235B-A22B | coding agent (open weights) | **0.0247** | 0.0217 | 0.0277 | 70 |
| 32 | OpenCode × Qwen3-32B | coding agent (open weights) | **0.0044** | 0.0059 | 0.0030 | 25 |
| 33 | OpenCode × Llama 4 Maverick (Converse proxy) | coding agent (open weights) | **0.0011** | 0.0005 | 0.0016 | 7 |
| 34 | OpenCode × Gemma 3 27B | coding agent (open weights) | 0.0000 | 0.0000 | 0.0000 | 0 |
| 35 | OpenCode × Gemma 3 12B | coding agent (open weights) | 0.0000 | 0.0000 | 0.0000 | 0 |
| 36 | OpenCode × Magistral Small | coding agent (open weights) | 0.0000 | 0.0000 | 0.0000 | 0 |
| 37 | OpenCode × Llama 3.3 70B (Converse proxy) | coding agent (open weights) | 0.0000 | 0.0000 | 0.0000 | 0 |

DeepSeek R1 is not ranked: Bedrock offers no tool use for it in either streaming or non-streaming mode, so
it cannot act as a coding agent (`bench/roster_bedrock_open.json`, route `unsupported`).
