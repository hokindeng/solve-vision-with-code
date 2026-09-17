#!/usr/bin/env python3
"""Move each colored object to the same-colored star marker along a straight path."""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48


def main():
    img = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = img.shape
    bg_color = np.array([255, 255, 255], dtype=np.uint8)
    mask = (img != bg_color).any(axis=2)
    labels, n = ndimage.label(mask)

    stars, objects = [], []
    for i in range(1, n + 1):
        m = labels == i
        ys, xs = np.where(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        fill = len(xs) / ((x1 - x0 + 1) * (y1 - y0 + 1))
        color = tuple(int(c) for c in np.median(img[m], axis=0))
        center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        comp = dict(mask=m, color=color, center=center, bbox=(x0, x1, y0, y1))
        # 4-point stars are sparse in their bounding box; objects are dense.
        (stars if fill < 0.4 else objects).append(comp)

    # Background with objects removed (stars stay in place).
    base = img.copy()
    for o in objects:
        base[o["mask"]] = bg_color

    # Prepare object sprites and their target positions.
    moves = []
    for o in objects:
        x0, x1, y0, y1 = o["bbox"]
        sprite = img[y0:y1 + 1, x0:x1 + 1].copy()
        smask = o["mask"][y0:y1 + 1, x0:x1 + 1]
        cands = [s for s in stars if np.abs(np.array(s["color"]) - np.array(o["color"])).sum() < 60]
        if not cands:
            raise RuntimeError(f"no star for color {o['color']}")
        target = min(cands, key=lambda s: np.linalg.norm(s["center"] - o["center"]))
        moves.append(dict(sprite=sprite, smask=smask, tl=np.array([x0, y0], dtype=float),
                          delta=target["center"] - o["center"]))

    frames = []
    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1)
        frame = base.copy()
        for mv in moves:
            tl = np.round(mv["tl"] + t * mv["delta"]).astype(int)
            x, y = tl
            h, w = mv["smask"].shape
            region = frame[y:y + h, x:x + w]
            region[mv["smask"]] = mv["sprite"][mv["smask"]]
        frames.append(frame)
    frames[0] = img.copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
