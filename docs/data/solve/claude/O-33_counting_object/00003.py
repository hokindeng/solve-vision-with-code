#!/usr/bin/env python3
"""Count the objects in first_frame.png, highlighting each border in turn,
then display 'Count: N' in the centre. Writes /app/output/video.mp4."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 30, 16
HIGHLIGHT = np.array([255, 200, 0], dtype=np.uint8)   # amber ring
RING = 5                                              # ring thickness (px)


def detect_objects(img):
    """Return list of (mask, cx, cy) for every filled+outlined shape."""
    h, w, _ = img.shape
    # Background = most common colour (border pixels are a safe sample too).
    flat = img.reshape(-1, 3)
    cols, cnt = np.unique(flat, axis=0, return_counts=True)
    bg = cols[np.argmax(cnt)]
    fg = (np.abs(img.astype(int) - bg.astype(int)).sum(axis=2) > 30).astype(np.uint8)
    # Close tiny gaps so the outline+fill form a single component.
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
    objs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 40:      # ignore specks
            continue
        objs.append(((lab == i), cent[i][0], cent[i][1]))
    # Systematic order: top-to-bottom in rows, left-to-right within a row.
    objs.sort(key=lambda o: (round(o[2] / 120), o[1]))
    return objs


def ring_mask(mask):
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING + 1, 2 * RING + 1))
    dil = cv2.dilate(m, k)
    return (dil > 0) & (m == 0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    objs = detect_objects(base)
    n_obj = len(objs)
    rings = [ring_mask(o[0]) for o in objs]

    # Timeline: frame 0 untouched; counting phase; then text phase.
    text_start = 22                      # frames 22..29 show the count
    count_frames = text_start - 1        # frames 1..21
    per = count_frames / max(n_obj, 1)

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    label = f"Count: {n_obj}"

    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        if f >= 1:
            # Objects whose counting has begun get (and keep) a highlighted border.
            for i, r in enumerate(rings):
                start = 1 + i * per
                if f >= start:
                    a = min(1.0, (f - start + 1) / 3.0)   # quick fade-in
                    img[r] = (a * HIGHLIGHT + (1 - a) * img[r]).astype(np.uint8)
        if f >= text_start:
            pil = Image.fromarray(img)
            d = ImageDraw.Draw(pil)
            bbox = d.textbbox((0, 0), label, font=font, anchor="lt")
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = 512 - tw / 2 - bbox[0], 512 - th / 2 - bbox[1]
            a = min(1.0, (f - text_start + 1) / 3.0)
            layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).text((x, y), label, font=font,
                                       fill=(0, 0, 0, int(255 * a)), anchor="lt")
            pil = Image.alpha_composite(pil.convert("RGBA"), layer).convert("RGB")
            img = np.array(pil)
        frames.append(img)

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-preset", "slow", "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print(f"objects: {n_obj}; wrote {OUT}")


if __name__ == "__main__":
    main()
