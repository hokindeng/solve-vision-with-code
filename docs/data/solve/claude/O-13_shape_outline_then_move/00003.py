#!/usr/bin/env python3
"""Animate the plus: filled -> outline-only (step 1), then move down 100 px (step 2)."""
import os, subprocess
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS, SS = 64, 16, 4

YELLOW = np.array([229, 229, 45], dtype=np.float32)
# Plus geometry measured from first_frame.png (pixels 108..268 x 602..762, arm width 53)
CX, CY = 188.5, 682.5          # centre in continuous pixel coords
HL, HW = 80.5, 26.5            # half length of arms, half width of arms
STROKE_OUT, STROKE_IN = 3.0, 2.0   # outline: 3 px outside the fill edge, 2 px inside (5 px total, as the reference rect)
MOVE = 100.0                   # "huge" downward move, same as the reference rectangle (282 -> 382)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def plus_mask(xs, ys, cx, cy, e):
    """Plus dilated by e (negative = eroded). xs, ys broadcastable grids."""
    if HW + e <= 0:
        return np.zeros(np.broadcast(xs, ys).shape, bool)
    dx, dy = np.abs(xs - cx), np.abs(ys - cy)
    return ((dx <= HW + e) & (dy <= HL + e)) | ((dy <= HW + e) & (dx <= HL + e))


def render_shape(h, w, cy, outer, inner):
    """Anti-aliased coverage of (plus+outer) minus (plus-inner), inner=None -> solid."""
    off = (np.arange(SS) + 0.5) / SS
    y0, y1 = int(cy - HL - outer) - 2, int(cy + HL + outer) + 3
    x0, x1 = int(CX - HL - outer) - 2, int(CX + HL + outer) + 3
    ys = (np.arange(y0, y1)[:, None] + off[None, :]).reshape(-1)[:, None]
    xs = (np.arange(x0, x1)[:, None] + off[None, :]).reshape(-1)[None, :]
    m = plus_mask(xs, ys, CX, cy, outer)
    if inner is not None:
        m &= ~plus_mask(xs, ys, CX, cy, -inner)
    cov = m.reshape(y1 - y0, SS, x1 - x0, SS).mean(axis=(1, 3))
    full = np.zeros((h, w), np.float32)
    full[y0:y1, x0:x1] = cov
    return full


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
    h, w, _ = base.shape
    # erase the original plus (exact solid fill) -> background
    erase = render_shape(h, w, CY, 0.0, None) > 0
    base[erase] = 255.0

    # timeline (frames): hold, step 1, hold, step 2, hold
    S1A, S1B, S2A, S2B = 4, 30, 35, 60
    frames = []
    for i in range(N_FRAMES):
        s1 = ease((i - S1A) / (S1B - S1A))
        s2 = ease((i - S2A) / (S2B - S2A))
        outer = STROKE_OUT * s1
        inner = None if s1 <= 0 else HW - (HW - STROKE_IN) * s1
        cy = CY + MOVE * s2
        a = render_shape(h, w, cy, outer, inner)[..., None]
        img = base * (1 - a) + YELLOW * a
        frames.append(np.clip(img + 0.5, 0, 255).astype(np.uint8))

    # sanity: frame 0 must equal first frame exactly
    orig = np.array(Image.open(FIRST).convert("RGB"))
    assert np.array_equal(frames[0], orig), "frame 0 differs from first_frame.png"

    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
