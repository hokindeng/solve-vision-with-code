#!/usr/bin/env python3
"""Count the objects in first_frame.png and render the counting animation."""
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 55
HIGHLIGHT = (230, 30, 30)  # RGB


def detect_objects(img):
    """Return list of (contour, bbox) for each non-background component."""
    bg = img[0, 0].astype(int)
    diff = np.abs(img.astype(int) - bg).sum(axis=2)
    mask = (diff > 30).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        objs.append((cnt, (x, y, w, h)))
    # reading order: top-to-bottom, then left-to-right (row bands of 120px)
    objs.sort(key=lambda o: ((o[1][1] + o[1][3] // 2) // 120, o[1][0] + o[1][2] // 2))
    return objs


def load_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W = base.shape[:2]
    objs = detect_objects(base)
    count = len(objs)

    # Schedule: frame 0 untouched; counting frames; then the final text.
    text_start = 43
    count_frames = list(range(1, text_start))
    per = len(count_frames) / count  # frames per object

    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        if f >= 1:
            # how many objects have been counted (highlighted) by this frame
            done = min(count, int((f - 1) / per) + 1)
            for k in range(done):
                cnt, _ = objs[k]
                age = (f - 1) - int(k * per)
                thick = 7 if age == 0 else (5 if age == 1 else 4)
                cv2.drawContours(img, [cnt], -1, HIGHLIGHT, thick, lineType=cv2.LINE_AA)
        if f >= text_start:
            pil = Image.fromarray(img)
            d = ImageDraw.Draw(pil)
            font = load_font(72)
            txt = f"Count: {count}"
            bbox = d.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (W - tw) // 2 - bbox[0]
            y = (H - th) // 2 - bbox[1]
            d.text((x, y), txt, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
            img = np.array(pil)
        frames.append(img)

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
        OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"objects counted: {count}; wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
