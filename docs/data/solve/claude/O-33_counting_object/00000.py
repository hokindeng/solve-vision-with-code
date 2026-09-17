#!/usr/bin/env python3
"""Count the geometric objects in first_frame.png and render an animation
that highlights each object's border in turn, then displays 'Count: N'."""
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 70
W = H = 1024

base = np.array(Image.open(SRC).convert("RGB"))
bg = np.array([240, 240, 240], dtype=np.uint8)

# --- detect objects: everything that is not background, grouped into components
mask = (np.abs(base.astype(int) - bg.astype(int)).sum(axis=2) > 30).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
objs = []
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < 80:
        continue
    objs.append((i, cents[i]))
# systematic order: rows from top to bottom (band of ~170px), left to right within a row
objs.sort(key=lambda o: (int(o[1][1] // 170), o[1][0]))
count = len(objs)
print("objects found:", count)

# border highlight rings (outline region, slightly thickened outward)
rings = []
for i, _ in objs:
    m = (labels == i).astype(np.uint8)
    outer = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=3)
    inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)
    rings.append((outer - inner).astype(bool))

# --- timeline
HOLD_START = 2                       # untouched frames at the beginning
END_FRAMES = 14                      # frames showing the final count
count_frames = N_FRAMES - HOLD_START - END_FRAMES
per = count_frames / count           # frames per object (fractional)

HIGHLIGHT = np.array([255, 0, 0], dtype=np.uint8)

try:
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except OSError:
    font_big = font_small = ImageFont.load_default()


def label_frame(img, k, cx, cy):
    """Draw the running number next to the object being counted."""
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    txt = str(k)
    x, y = int(cx), int(cy)
    d.text((x, y), txt, fill=(200, 0, 0), font=font_small, anchor="mm",
           stroke_width=2, stroke_fill=(255, 255, 255))
    return np.array(pil)


def final_frame(img, alpha=1.0):
    pil = Image.fromarray(img)
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    txt = f"Count: {count}"
    d.text((W // 2, H // 2), txt, fill=(0, 0, 0, int(255 * alpha)), font=font_big, anchor="mm",
           stroke_width=4, stroke_fill=(255, 255, 255, int(255 * alpha)))
    pil = Image.alpha_composite(pil.convert("RGBA"), overlay)
    return np.array(pil.convert("RGB"))


frames = []
for f in range(N_FRAMES):
    img = base.copy()
    if f < HOLD_START:
        pass
    elif f < HOLD_START + count_frames:
        k = min(int((f - HOLD_START) / per), count - 1)   # index of object being counted
        img[rings[k]] = HIGHLIGHT
        cx, cy = objs[k][1]
        img = label_frame(img, k + 1, cx, cy)
    else:
        t = f - (HOLD_START + count_frames)
        alpha = min(1.0, (t + 1) / 3.0)
        img = final_frame(img, alpha)
    frames.append(img)

# --- encode with ffmpeg (H.264, yuv420p)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(np.ascontiguousarray(fr).tobytes())
p.stdin.close()
p.wait()
print("wrote", OUT, "frames:", len(frames))
