#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: group shapes by type, sort by size, line them up."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 96
HOLD_START, HOLD_END = 8, 12  # frames held still at start / end


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = img.shape

    # Background is uniform: take the most common colour.
    cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    bg = cols[counts.argmax()]
    fg = (np.abs(img.astype(int) - bg.astype(int)).sum(2) > 0).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        m = (lab[y:y + h, x:x + w] == i)
        fill = area / float(w * h)
        kind = "square" if fill > 0.85 else "triangle"
        shapes.append(dict(x=x, y=y, w=w, h=h, size=max(w, h), kind=kind,
                           sprite=img[y:y + h, x:x + w].copy(), mask=m))

    # Group order: group whose shapes currently sit further left goes first.
    kinds = sorted({s["kind"] for s in shapes},
                   key=lambda k: np.mean([s["x"] + s["w"] / 2 for s in shapes if s["kind"] == k]))
    ordered = []
    for k in kinds:
        ordered += sorted([s for s in shapes if s["kind"] == k], key=lambda s: s["size"])

    # Target layout: one horizontal row, vertically centred, equal gaps.
    total_w = sum(s["w"] for s in ordered)
    gap = (W - total_w) / (len(ordered) + 1)
    cx = gap
    for s in ordered:
        s["tx"], s["ty"] = cx, (H - s["h"]) / 2
        s["sx"], s["sy"] = float(s["x"]), float(s["y"])
        cx += s["w"] + gap

    background = np.empty_like(img)
    background[:] = bg

    def ease(t):
        return 0.5 - 0.5 * np.cos(np.pi * t)

    def render(t):
        frame = background.copy()
        # Draw larger shapes first so smaller ones stay visible if paths cross.
        for s in sorted(ordered, key=lambda s: -s["size"]):
            px = int(round(s["sx"] + (s["tx"] - s["sx"]) * t))
            py = int(round(s["sy"] + (s["ty"] - s["sy"]) * t))
            h, w = s["h"], s["w"]
            roi = frame[py:py + h, px:px + w]
            roi[s["mask"]] = s["sprite"][s["mask"]]
        return frame

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    move = N_FRAMES - HOLD_START - HOLD_END
    for f in range(N_FRAMES):
        if f < HOLD_START:
            frame = img
        else:
            t = min(1.0, (f - HOLD_START) / float(move - 1))
            frame = render(ease(t))
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    Image.fromarray(render(1.0)).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
