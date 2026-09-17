#!/usr/bin/env python3
"""Complete the Raven-style matrix: draw the missing orange square in the
bottom-right cell, animated over 35 frames at 16 fps.

Rule: every row/column contains one orange square, one red diamond and one
brown notched shape, so the bottom-right cell needs the orange square.
"""
import os
import subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 35, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# --- reference square: mid-left cell, exact pixel mask ---------------------
ORANGE = np.array([255, 140, 0], dtype=np.uint8)
ref = base[453:569, 112:228]
ref_mask = np.all(ref == ORANGE, axis=2)          # 116x116 boolean stroke
assert ref_mask.sum() > 0

# --- target cell (bottom-right): same offsets relative to cell centre ------
# grid lines: cols 682/1022, rows 683/1021 -> cell centre (852, 852)
TY0, TX0 = 794, 794                               # top-left of 116x116 box
target_mask = np.zeros((H, W), dtype=bool)
target_mask[TY0:TY0 + 116, TX0:TX0 + 116] = ref_mask

# --- question-mark placeholder in the bottom-right cell --------------------
cell = base[684:1021, 684:1021]
q_local = np.any(cell != 255, axis=2)
q_mask = np.zeros((H, W), dtype=bool)
q_mask[684:1021, 684:1021] = q_local

# --- parametrise the stroke by position along the perimeter (clockwise) ----
ys, xs = np.nonzero(target_mask)
cy, cx = TY0 + 57.5, TX0 + 57.5
# angle from "up", clockwise, starting at the top-left corner
ang = np.arctan2(xs - cx, -(ys - cy))             # 0 at top, +clockwise
t = (ang + np.pi * 0.75) % (2 * np.pi) / (2 * np.pi)   # 0 at top-left corner

FADE_END = 8                                      # frames 1..8 fade the "?"
DRAW_START, DRAW_END = 9, N_FRAMES - 1            # frames 9..34 draw square

def make_frame(i):
    f = base.copy()
    if i >= 1:
        a = min(1.0, i / FADE_END)                # fade "?" toward white
        px = f[q_mask].astype(np.float32)
        f[q_mask] = np.clip(px + (255 - px) * a, 0, 255).astype(np.uint8)
    if i >= DRAW_START:
        p = (i - DRAW_START + 1) / (DRAW_END - DRAW_START + 1)
        sel = t <= p + 1e-9
        f[ys[sel], xs[sel]] = ORANGE
    return f

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [make_frame(i) for i in range(N_FRAMES)]
    assert np.array_equal(frames[0], base)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
           "-preset", "slow", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, "ffmpeg failed"
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
