#!/usr/bin/env python3
"""Sort scattered animal faces by size (largest -> smallest) and line them up
on the bottom baseline. Everything else in the frame is left untouched."""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 40, 16
GAP = 60  # horizontal spacing between sprites in the final row


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = img.shape
    bg = img[0, 0].astype(int)
    diff = (np.abs(img.astype(int) - bg).sum(-1) > 0)

    # Baseline: long horizontal run(s) of non-background pixels.
    row_counts = diff.sum(1)
    base_rows = np.where(row_counts > W // 2)[0]
    baseline_top = int(base_rows.min())
    sprite_mask = diff.copy()
    sprite_mask[base_rows] = False  # exclude the baseline from sprite detection

    # Merge sub-parts of each face (eyes, ears...) into one blob per animal.
    closed = ndimage.binary_closing(sprite_mask, structure=np.ones((9, 9)), iterations=2)
    closed = ndimage.binary_fill_holes(closed)
    closed[base_rows] = False
    lab, n = ndimage.label(closed)

    sprites = []
    for i in range(1, n + 1):
        comp = lab == i
        raw = comp & sprite_mask
        ys, xs = np.where(raw)
        if len(ys) < 50:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        m = comp[y0:y1 + 1, x0:x1 + 1]
        sprites.append({
            "pix": img[y0:y1 + 1, x0:x1 + 1].copy(),
            "mask": m,
            "x0": int(x0), "y0": int(y0),
            "w": int(x1 - x0 + 1), "h": int(y1 - y0 + 1),
            "size": int(m.sum()),
        })

    # Largest to smallest, laid out left to right, bottoms on the baseline.
    sprites.sort(key=lambda s: -s["size"])
    total = sum(s["w"] for s in sprites) + GAP * (len(sprites) - 1)
    x = (W - total) // 2
    for s in sprites:
        s["tx"], s["ty"] = x, baseline_top - s["h"]
        x += s["w"] + GAP

    # Background = original frame with the sprites removed.
    background = img.copy()
    for s in sprites:
        y0, x0, h, w = s["y0"], s["x0"], s["h"], s["w"]
        background[y0:y0 + h, x0:x0 + w][s["mask"]] = bg

    def ease(t):
        return 0.5 - 0.5 * np.cos(np.pi * t)

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        t = ease(f / (N_FRAMES - 1))
        frame = background.copy()
        for s in sprites:
            cx = int(round(s["x0"] + (s["tx"] - s["x0"]) * t))
            cy = int(round(s["y0"] + (s["ty"] - s["y0"]) * t))
            h, w = s["h"], s["w"]
            region = frame[cy:cy + h, cx:cx + w]
            region[s["mask"]] = s["pix"][s["mask"]]
        frames.append(frame)
    frames[0] = img  # first frame is exactly the source image

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT, "with", len(frames), "frames;",
          [(s["size"], s["tx"], s["ty"]) for s in sprites])


if __name__ == "__main__":
    main()
