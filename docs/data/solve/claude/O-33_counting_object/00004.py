#!/usr/bin/env python3
"""Generate the counting video: highlight each object's border in turn, then show 'Count: N'."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 40
HIGHLIGHT = (255, 200, 0)  # RGB, amber highlight ring
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def detect_objects(img):
    """Return list of (mask, cx, cy) for every object, ordered top-to-bottom, left-to-right."""
    bg = img[0, 0].astype(int)
    diff = np.abs(img.astype(int) - bg).sum(axis=2)
    mask = (diff > 30).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        m = (labels == i).astype(np.uint8)
        # fill any interior holes so the ring hugs the outer outline only
        h, w = m.shape
        ff = m.copy()
        flood = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, flood, (0, 0), 1)
        holes = (ff == 0).astype(np.uint8)
        m = m | holes
        objs.append((m, cents[i][0], cents[i][1]))
    # raster-style scan order: rows of ~200px, then by x
    objs.sort(key=lambda o: (int(o[2] // 200), o[1]))
    return objs


def ring_mask(mask, thickness):
    k = 2 * thickness + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask, kernel)
    return (dil > 0) & (mask == 0)


def draw_text(frame, text):
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_PATH, 72)
    W, H = pil.size
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - tw) // 2 - bbox[0]
    y = (H - th) // 2 - bbox[1]
    # subtle outline for legibility
    for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)]:
        d.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
    d.text((x, y), text, font=font, fill=(20, 20, 20))
    return np.array(pil)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    objs = detect_objects(base)
    n_obj = len(objs)

    # Timeline: frame 0 untouched; counting phase frames 1..COUNT_END; text phase after.
    COUNT_END = 30
    per = (COUNT_END - 1) / n_obj  # frames per object
    frames = []
    for f in range(N_FRAMES):
        frame = base.copy()
        if f >= 1:
            t = f - 1
            for idx, (m, cx, cy) in enumerate(objs):
                start = idx * per
                if t < start:
                    break
                # ring grows in over the first few frames of its slot, then stays
                prog = min(1.0, (t - start + 1) / max(1.0, per * 0.6))
                thick = 2 + int(round(4 * prog))
                ring = ring_mask(m, thick)
                frame[ring] = HIGHLIGHT
                # running count label next to the object
                label = str(idx + 1)
                pil = Image.fromarray(frame)
                d = ImageDraw.Draw(pil)
                font = ImageFont.truetype(FONT_PATH, 34)
                ys, xs = np.nonzero(m)
                lx = int(xs.max()) + 12
                ly = int(ys.min()) - 8
                if lx > 990:
                    lx = int(xs.min()) - 36
                if ly < 4:
                    ly = int(ys.max()) - 30
                d.text((lx, ly), label, font=font, fill=HIGHLIGHT,
                       stroke_width=2, stroke_fill=(0, 0, 0))
                frame = np.array(pil)
        if f >= COUNT_END + 1:
            frame = draw_text(frame, f"Count: {n_obj}")
        frames.append(frame)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print(f"objects: {n_obj}; wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
