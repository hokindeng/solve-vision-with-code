#!/usr/bin/env python3
"""Generate video: set odd-position lights on, even-position lights off."""
import os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, NFRAMES = 16, 35

def find_lights(im):
    """Return list of (x0, x1, cx) horizontal extents of non-white column runs."""
    mask = (im != 255).any(axis=2)
    cols = mask.any(axis=0)
    runs, s = [], None
    for x in range(len(cols)):
        if cols[x] and s is None: s = x
        if not cols[x] and s is not None: runs.append((s, x - 1)); s = None
    if s is not None: runs.append((s, len(cols) - 1))
    return runs

def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape
    runs = find_lights(base)
    assert len(runs) == 8, runs
    # Determine state of each light: gold fill present -> on
    states, centers = [], []
    for a, b in runs:
        cx = (a + b) // 2
        centers.append(cx)
        sub = base[:, a:b + 1]
        on = ((sub[..., 0] == 255) & (sub[..., 1] == 215) & (sub[..., 2] == 0)).any()
        states.append(bool(on))
    # Vertical extent shared by all lights
    mask = (base != 255).any(axis=2)
    ys = np.where(mask.any(axis=1))[0]
    y0, y1 = ys.min() - 4, ys.max() + 5
    # Sprite half-width: use the largest run (the glow) plus margin
    hw = max(b - a for a, b in runs) // 2 + 4

    def patch(cx):
        return base[y0:y1, cx - hw:cx + hw + 1].astype(np.float32)

    on_idx = states.index(True); off_idx = states.index(False)
    on_sprite, off_sprite = patch(centers[on_idx]), patch(centers[off_idx])
    # Sanity: every light lies fully inside its sprite window, and windows don't overlap
    for (a, b), cx in zip(runs, centers):
        assert cx - hw <= a and b <= cx + hw, (a, b, cx, hw)
    for c0, c1 in zip(centers, centers[1:]):
        assert c0 + hw < c1 - hw
    # Sprite borders must be pure white so they blend seamlessly into the background
    for sp in (on_sprite, off_sprite):
        assert (sp[:, 0] == 255).all() and (sp[:, -1] == 255).all()
        assert (sp[0] == 255).all() and (sp[-1] == 255).all()

    target = [(i % 2 == 0) for i in range(8)]  # index 0 = position 1 (odd) -> on
    changing = [i for i in range(8) if states[i] != target[i]]

    # Schedule: staggered crossfades spanning frames 1..NFRAMES-1
    n = len(changing)
    fade_len = 6  # frames per transition
    total = NFRAMES - 1
    starts = [1 + round(k * (total - fade_len) / max(n - 1, 1)) for k in range(n)] if n > 1 else [1]

    tmp = tempfile.mkdtemp()
    for f in range(NFRAMES):
        frame = base.copy()
        for k, i in enumerate(changing):
            t = np.clip((f - starts[k] + 1) / fade_len, 0.0, 1.0)
            if f == NFRAMES - 1: t = 1.0
            t = t * t * (3 - 2 * t)  # smoothstep
            cx = centers[i]
            if t <= 0.0:
                continue  # untouched: keep the original pixels exactly
            src = patch(cx)  # this light's own original rendering
            dst = off_sprite if states[i] else on_sprite
            blend = (1 - t) * src + t * dst
            frame[y0:y1, cx - hw:cx + hw + 1] = np.clip(np.rint(blend), 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(os.path.join(tmp, f"f{f:04d}.png"))

    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT, "changing lights (1-based):", [i + 1 for i in changing])

if __name__ == "__main__":
    main()
