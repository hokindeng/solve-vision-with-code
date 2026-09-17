#!/usr/bin/env python3
"""Identify the unique shape in first_frame.png and animate circling it in red."""
import math
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess, os, shutil

BASE = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 60
FPS = 16
SCALE = 4  # supersampling for anti-aliased red circle

def find_shapes(img):
    """Return list of dicts with bbox, area, centroid and a shape descriptor."""
    arr = np.array(img.convert("RGB")).astype(int)
    bg = arr[0, 0]
    mask = (np.abs(arr - bg).sum(axis=2) > 30).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = max(cv2.contourArea(hull), 1)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        shapes.append(dict(
            bbox=(x, y, w, h), area=int(area), cx=cents[i][0], cy=cents[i][1],
            fill=area / (w * h),                # bbox fill ratio (square≈1, circle≈0.785, tri≈0.5)
            solidity=area / hull_area,
            circularity=4 * math.pi * area / max(peri * peri, 1),
            nverts=len(approx),
            color=tuple(int(v) for v in arr[labels == i].mean(axis=0)),
        ))
    return shapes

def feature_vec(s):
    return np.array([s["fill"] * 4, s["circularity"] * 4, s["solidity"] * 4,
                     min(s["nverts"], 12) / 3.0,
                     math.sqrt(s["area"]) / 40.0,
                     *(np.array(s["color"]) / 255.0)])

def pick_unique(shapes):
    """The unique shape is the one farthest from the others in feature space."""
    F = np.array([feature_vec(s) for s in shapes])
    best, best_score = None, -1
    for i in range(len(shapes)):
        others = np.delete(F, i, axis=0)
        # distance from this shape to the median of the rest, relative to the spread of the rest
        med = np.median(others, axis=0)
        spread = np.abs(others - med).sum(axis=1).mean() + 1e-3
        score = np.abs(F[i] - med).sum() / spread
        if score > best_score:
            best, best_score = i, score
    return shapes[best]

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)

def main():
    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    shapes = find_shapes(base)
    target = pick_unique(shapes)
    x, y, w, h = target["bbox"]
    cx, cy = x + w / 2.0, y + h / 2.0
    radius = 0.5 * math.hypot(w, h) + 18
    width = 6

    # Timeline: hold (identification pause) -> draw arc -> hold finished result
    hold_start = 10
    draw_frames = 38
    frames = []
    for f in range(N_FRAMES):
        if f < hold_start:
            prog = 0.0
        else:
            prog = min(1.0, (f - hold_start) / (draw_frames - 1))
        prog = ease(prog)
        frame = base.copy()
        if prog > 0:
            # draw arc on transparent supersampled overlay, then composite
            ov = Image.new("RGBA", (W * SCALE, H * SCALE), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            r = radius * SCALE
            box = [(cx - radius) * SCALE, (cy - radius) * SCALE,
                   (cx + radius) * SCALE, (cy + radius) * SCALE]
            start = -90
            end = start + 360 * prog
            if prog >= 0.999:
                d.ellipse(box, outline=(220, 30, 30, 255), width=width * SCALE)
            else:
                d.arc(box, start=start, end=end, fill=(220, 30, 30, 255), width=width * SCALE)
                # round caps
                for ang in (start, end):
                    a = math.radians(ang)
                    px = (cx + radius * math.cos(a)) * SCALE
                    py = (cy + radius * math.sin(a)) * SCALE
                    hw = width * SCALE / 2
                    d.ellipse([px - hw, py - hw, px + hw, py + hw], fill=(220, 30, 30, 255))
            ov = ov.resize((W, H), Image.LANCZOS)
            frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
        frames.append(np.array(frame))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"unique shape: bbox={target['bbox']} fill={target['fill']:.3f} "
          f"circ={target['circularity']:.3f} nverts={target['nverts']}")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
