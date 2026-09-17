import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80
RED = (220, 30, 30)

def detect_hollow(img):
    gray = np.array(img.convert("L"))
    dark = (gray < 128).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    hollow = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 10 or h < 10:
            continue
        r = (w + h) / 4.0
        disk_area = math.pi * r * r
        fill = area / disk_area
        if fill < 0.5:  # thin ring -> hollow point
            cx, cy = x + w / 2.0, y + h / 2.0
            hollow.append((cx, cy, r))
    hollow.sort(key=lambda c: (c[1], c[0]))
    return hollow

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

def draw_arc_ring(draw, cx, cy, R, frac, width):
    if frac <= 0:
        return
    bbox = [cx - R, cy - R, cx + R, cy + R]
    if frac >= 1:
        draw.ellipse(bbox, outline=RED, width=width)
    else:
        draw.arc(bbox, start=-90, end=-90 + 360 * frac, fill=RED, width=width)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    hollow = detect_hollow(base)
    print("hollow points:", [(round(x), round(y), round(r)) for x, y, r in hollow])

    k = len(hollow)
    # schedule: hold 8 frames, then each ring drawn over a slot, then hold to end
    hold_start, hold_end = 8, 12
    active = N_FRAMES - hold_start - hold_end
    slot = active / max(k, 1)
    draw_len = slot * 0.85
    width = 5

    frames = []
    for f in range(N_FRAMES):
        im = base.copy()
        d = ImageDraw.Draw(im)
        for i, (cx, cy, r) in enumerate(hollow):
            t0 = hold_start + i * slot
            frac = ease((f - t0) / draw_len)
            draw_arc_ring(d, cx, cy, r + 12, frac, width)
        frames.append(im)

    frames[0] = base.copy()  # first frame must be untouched
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "16", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
