# Paper tables (generated paper_tables.py)

## Table 1 — leaderboard

| # | model | kind | Overall | ID | OOD |
|---|---|---|---|---|---|
| 1 | Codex/gpt-6-astra | coding agent | 0.923 | 0.879 | 0.967 |
| 2 | Claude Code/Fable 5.1 | coding agent | 0.894 | 0.830 | 0.958 |
| 3 | Gemini CLI/3.1 Pro | coding agent | 0.893 | 0.826 | 0.960 |
| 4 | VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model) | video model | 0.670 | 0.808 | 0.532 |
| 5 | GLM-5 | coding agent (open weights) | 0.572 | 0.479 | 0.664 |
| 6 | VBVR-Pro-Wan2.1-I2V-14B | video model | 0.562 | 0.730 | 0.395 |
| 7 | Kimi K2.5 | coding agent (open weights) | 0.557 | 0.454 | 0.660 |
| 8 | VBVR-Wan2.2 | video model | 0.517 | 0.548 | 0.486 |
| 9 | Seedance 2.0 | video model | 0.499 | 0.451 | 0.547 |
| 10 | VBVR-Pro-Wan2.2-TI2V-5B | video model | 0.470 | 0.641 | 0.300 |
| 11 | MiniMax M2.5 | coding agent (open weights) | 0.451 | 0.386 | 0.515 |
| 12 | VBVR-Pro-LTX2.3 | video model | 0.425 | 0.527 | 0.324 |
| 13 | Kling VIDEO 3.0 | video model | 0.392 | 0.356 | 0.427 |
| 14 | DeepSeek V3.2 | coding agent (open weights) | 0.378 | 0.303 | 0.452 |
| 15 | Devstral 2 123B | coding agent (open weights) | 0.350 | 0.275 | 0.426 |
| 16 | Kimi K2 Thinking | coding agent (open weights) | 0.313 | 0.218 | 0.407 |
| 17 | Veo 3.1 | video model | 0.309 | 0.312 | 0.305 |
| 18 | gpt-oss-120b | coding agent (open weights) | 0.270 | 0.196 | 0.344 |
| 19 | Qwen3-Coder-Next | coding agent (open weights) | 0.264 | 0.218 | 0.310 |
| 20 | Nemotron Super 3 120B | coding agent (open weights) | 0.200 | 0.139 | 0.261 |
| 21 | Wan2.2-I2V-A14B | video model | 0.182 | 0.157 | 0.207 |
| 22 | Qwen3-Coder-30B-A3B | coding agent (open weights) | 0.143 | 0.099 | 0.186 |
| 23 | Mistral Large 3 675B | coding agent (open weights) | 0.127 | 0.079 | 0.176 |
| 24 | LTX-2.3-I2AV | video model | 0.112 | 0.106 | 0.119 |
| 25 | GLM-4.7-Flash | coding agent (open weights) | 0.104 | 0.081 | 0.127 |
| 26 | Wan2.1-I2V-14B-720P | video model | 0.100 | 0.105 | 0.095 |
| 27 | gpt-oss-20b | coding agent (open weights) | 0.079 | 0.075 | 0.083 |
| 28 | Qwen3-Next-80B-A3B | coding agent (open weights) | 0.070 | 0.034 | 0.105 |
| 29 | Llama 4 Scout | coding agent (open weights) | 0.049 | 0.040 | 0.058 |
| 30 | Nemotron Nano 3 30B | coding agent (open weights) | 0.028 | 0.026 | 0.029 |
| 31 | Qwen3-VL-235B-A22B | coding agent (open weights) | 0.025 | 0.022 | 0.028 |
| 32 | Qwen3-32B | coding agent (open weights) | 0.004 | 0.006 | 0.003 |
| 33 | Llama 4 Maverick | coding agent (open weights) | 0.001 | 0.001 | 0.002 |
| 34 | OpenCode × Gemma 3 27B (open weights, Bedrock) | coding agent (open weights) | 0.000 | 0.000 | 0.000 |
| 35 | OpenCode × Gemma 3 12B (open weights, Bedrock) | coding agent (open weights) | 0.000 | 0.000 | 0.000 |
| 36 | OpenCode × Magistral Small (open weights, Bedrock) | coding agent (open weights) | 0.000 | 0.000 | 0.000 |
| 37 | OpenCode × Llama 3.3 70B (open weights, Bedrock via converse proxy) | coding agent (open weights) | 0.000 | 0.000 | 0.000 |

## Table 2 — by category (coding agents)

| model | Abstraction | Perception | Spatiality | Transformation | Knowledge | overall |
|---|---|---|---|---|---|---|
| Codex/gpt-6-astra | 0.957 | 0.953 | 0.903 | 0.902 | 0.863 | 0.923 |
| Claude Code/Fable 5.1 | 0.838 | 0.957 | 0.908 | 0.943 | 0.834 | 0.894 |
| Gemini CLI/3.1 Pro | 0.878 | 0.972 | 0.816 | 0.894 | 0.861 | 0.893 |
| GLM-5 | 0.478 | 0.748 | 0.491 | 0.569 | 0.506 | 0.572 |
| Kimi K2.5 | 0.427 | 0.727 | 0.527 | 0.582 | 0.490 | 0.557 |
| MiniMax M2.5 | 0.347 | 0.633 | 0.412 | 0.439 | 0.358 | 0.451 |
| DeepSeek V3.2 | 0.308 | 0.538 | 0.358 | 0.288 | 0.307 | 0.378 |
| Devstral 2 123B | 0.290 | 0.478 | 0.382 | 0.296 | 0.248 | 0.350 |
| Kimi K2 Thinking | 0.231 | 0.476 | 0.359 | 0.195 | 0.216 | 0.313 |
| gpt-oss-120b | 0.224 | 0.348 | 0.300 | 0.167 | 0.255 | 0.270 |
| Qwen3-Coder-Next | 0.260 | 0.364 | 0.195 | 0.180 | 0.232 | 0.264 |
| Nemotron Super 3 120B | 0.193 | 0.249 | 0.269 | 0.062 | 0.167 | 0.200 |
| Qwen3-Coder-30B-A3B | 0.183 | 0.123 | 0.171 | 0.067 | 0.143 | 0.143 |
| Mistral Large 3 675B | 0.075 | 0.200 | 0.148 | 0.057 | 0.118 | 0.127 |
| GLM-4.7-Flash | 0.158 | 0.097 | 0.088 | 0.057 | 0.086 | 0.104 |
| gpt-oss-20b | 0.085 | 0.064 | 0.078 | 0.087 | 0.088 | 0.079 |
| Qwen3-Next-80B-A3B | 0.044 | 0.093 | 0.097 | 0.017 | 0.079 | 0.070 |
| Llama 4 Scout | 0.102 | 0.017 | 0.041 | 0.025 | 0.051 | 0.049 |
| Nemotron Nano 3 30B | 0.037 | 0.023 | 0.039 | 0.012 | 0.024 | 0.028 |
| Qwen3-VL-235B-A22B | 0.025 | 0.013 | 0.034 | 0.009 | 0.044 | 0.025 |
| Qwen3-32B | 0.002 | 0.003 | 0.013 | 0.003 | 0.003 | 0.004 |
| Llama 4 Maverick | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.001 |

## Table 3 — efficiency and failure anatomy

| model | score | produced/500 | tool-use rate | tool calls/attempt | timeouts | median s | input tok | output tok |
|---|---|---|---|---|---|---|---|---|
| Codex/gpt-6-astra | 0.923 | 500 | 1.00 | 3.2 | 0 | 22 | 56,774 | 1,165 |
| Claude Code/Fable 5.1 | 0.894 | 500 | 1.00 | 9.2 | 0 | 105 | 233,475 | 7,315 |
| Gemini CLI/3.1 Pro | 0.893 | 500 | 1.00 | 25.3 | 0 | 218 | 658,650 | 7,539 |
| GLM-5 | 0.572 | 495 | 1.00 | 48.0 | 12 | 378 | 1,568,595 | 16,295 |
| Kimi K2.5 | 0.557 | 498 | 0.99 | 41.9 | 0 | 336 | 1,213,119 | 19,179 |
| MiniMax M2.5 | 0.451 | 488 | 0.99 | 45.7 | 0 | 420 | 1,294,886 | 19,752 |
| DeepSeek V3.2 | 0.378 | 475 | 1.00 | 32.9 | 0 | 277 | 635,753 | 13,622 |
| Devstral 2 123B | 0.350 | 494 | 0.99 | 38.7 | 0 | 344 | 838,956 | 13,119 |
| Kimi K2 Thinking | 0.313 | 406 | 0.96 | 21.8 | 1 | 125 | 508,661 | 15,458 |
| gpt-oss-120b | 0.270 | 380 | 1.00 | 9.5 | 0 | 34 | 116,969 | 3,886 |
| Qwen3-Coder-Next | 0.264 | 498 | 1.00 | 47.7 | 93 | 455 | 1,817,146 | 19,192 |
| Nemotron Super 3 120B | 0.200 | 436 | 0.95 | 22.9 | 0 | 59 | 467,204 | 8,327 |
| Qwen3-Coder-30B-A3B | 0.143 | 480 | 0.98 | 12.5 | 0 | 50 | 146,392 | 4,325 |
| Mistral Large 3 675B | 0.127 | 254 | 0.98 | 7.6 | 0 | 26 | 80,956 | 1,756 |
| GLM-4.7-Flash | 0.104 | 470 | 1.00 | 27.7 | 1 | 41 | 655,689 | 7,611 |
| gpt-oss-20b | 0.079 | 148 | 0.94 | 9.7 | 0 | 55 | 135,344 | 7,409 |
| Qwen3-Next-80B-A3B | 0.070 | 167 | 0.99 | 6.1 | 0 | 25 | 61,249 | 1,466 |
| Llama 4 Scout | 0.049 | 269 | 1.00 | 7.5 | 1 | 15 | 75,450 | 1,175 |
| Nemotron Nano 3 30B | 0.028 | 91 | 0.35 | 8.1 | 0 | 5 | 158,506 | 2,541 |
| Qwen3-VL-235B-A22B | 0.025 | 70 | 1.00 | 3.1 | 0 | 9 | 26,215 | 501 |
| Qwen3-32B | 0.004 | 25 | 1.00 | 1.8 | 0 | 6 | 23,536 | 304 |
| Llama 4 Maverick | 0.001 | 7 | 0.98 | 2.1 | 0 | 6 | 21,448 | 218 |
| bedrock-deepseek-r1 | 0.000 | 0 | 0.00 | 0.0 | 0 | 6 | 0 | 0 |
| Gemma 3 12B | 0.000 | 0 | 0.00 | 0.0 | 0 | 17 | 1,799 | 332 |
| Gemma 3 27B | 0.000 | 0 | 0.00 | 0.0 | 0 | 31 | 1,831 | 526 |
| Llama 3.3 70B | 0.000 | 0 | 0.00 | 0.0 | 0 | 5 | 6,052 | 176 |
| Magistral Small | 0.000 | 0 | 0.00 | 0.0 | 0 | 5 | 7,618 | 597 |

## Table 4 — hardest 15 tasks for the closed-model agents

| task | split | Codex/gpt-6-astra | Claude Code/Fable 5.1 | Gemini CLI/3.1 Pro |
|---|---|---|---|---|
| G-43_understand_scene_structure | In_Domain | 0.21 | 0.00 | 0.77 |
| O-14_shape_scale_then_outline | In_Domain | 0.78 | 0.01 | 0.58 |
| G-29_chart_extreme_with_data | In_Domain | 0.59 | 0.57 | 0.32 |
| O-13_shape_outline_then_move | In_Domain | 0.94 | 0.01 | 0.59 |
| O-29_ballcolor | In_Domain | 0.73 | 0.41 | 0.49 |
| O-32_rolling_ball | In_Domain | 0.32 | 0.70 | 0.61 |
| O-21_construction_blueprint | In_Domain | 0.80 | 0.62 | 0.24 |
| O-55_rotation | In_Domain | 0.68 | 0.64 | 0.41 |
| O-15_ball_bounces_given_time | In_Domain | 0.68 | 0.55 | 0.54 |
| G-218_identify_largest_angle_in_triangle | Out_of_Domain | 0.20 | 0.60 | 1.00 |
| G-15_grid_avoid_obstacles | In_Domain | 0.41 | 0.81 | 0.61 |
| O-23_domino_chain_branch_path_prediction | In_Domain | 0.83 | 0.48 | 0.59 |
| G-39_attention_shift_different | In_Domain | 0.62 | 0.74 | 0.62 |
| O-31_ball_eating | In_Domain | 0.79 | 0.89 | 0.63 |
| O-19_mirror_reflection | In_Domain | 0.76 | 0.84 | 0.80 |
