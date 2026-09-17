# Qualitative listings (Section 5.4)

Trimmed `solve.py` programs written by Codex CLI (gpt-6-astra), reproduced from the agent's own event log
(`events.jsonl`, the `cat > /app/solve.py <<'PY'` heredoc). Only the imports and the ffmpeg encoder
setup are removed, long lines are wrapped to 56 columns and a few statements are joined or split; the
trimmed programs were executed against the same first frame and produce frames byte-identical to the originals.

| File | Lane / split / task / instance | Score | Tool calls | Wall time |
|---|---|---|---|---|
| `O-56_raven.py` | codex-gpt-6-astra / Out-of-Domain_50 / O-56_raven_data-generator / 00000 | 1.0 | 2 | 17 s |
| `G-13_grid_number_sequence.py` | codex-gpt-6-astra / In-Domain_50 / G-13_grid_number_sequence_data-generator / 00000 | 1.0 | 3 | 20 s |

Run logs: `bench/runs/<lane>/<split>/<task>/<idx>/app/events.jsonl` on the release bucket (see `DATA_MANIFEST.md`).
Included in the paper by `\verbatiminput` from `paper/sec/5_analysis.tex`.
