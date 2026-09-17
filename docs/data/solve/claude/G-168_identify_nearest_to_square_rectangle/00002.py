"""Generate video: compare rectangles' aspect ratios, circle the one closest to a square."""
import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N = 16, 48

base = Image.open(FIRST).convert("RGB")
arr = np.array(base)
W, H = base.size

# --- detect rectangles: connected components of non-white pixels ---
mask = np.any(arr < 250, axis=2)
mask = ndimage.binary_closing(mask, iterations=2)
labels, n = ndimage.label(mask)
rects = []
for i, sl in enumerate(ndimage.find_objects(labels), start=1):
    if sl is None:
        continue
    ys, xs = sl
    w, h = xs.stop - xs.start, ys.stop - ys.start
    if w < 10 or h < 10:
        continue
    ratio = w / h
    score = abs(math.log(ratio))  # 0 == perfect square
    rects.append(dict(x0=xs.start, y0=ys.start, x1=xs.stop - 1, y1=ys.stop - 1,
                      w=w, h=h, ratio=ratio, score=score))
rects.sort(key=lambda r: (r["y0"], r["x0"]))
best = min(rects, key=lambda r: r["score"])
for r in rects:
    print(f"rect at ({r['x0']},{r['y0']}) {r['w']}x{r['h']} ratio={r['ratio']:.3f} "
          f"{'<-- closest to square' if r is best else ''}")

# --- animation timeline ---
# Phase 1 (frames 1..COMPARE_END): inspect each rectangle in turn with a thin
# gray outline that fades in/out (temporary, disappears afterwards).
# Phase 2 (remaining frames): red circle grows/draws around the best rectangle.
COMPARE_END = 26
per = COMPARE_END // max(1, len(rects))

cx = (best["x0"] + best["x1"]) / 2
cy = (best["y0"] + best["y1"]) / 2
R = math.hypot(best["w"], best["h"]) / 2 + 14  # circle just outside the corners

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

frames = []
for f in range(N):
    img = base.copy()
    d = ImageDraw.Draw(img)
    if f == 0:
        frames.append(img); continue
    if f <= COMPARE_END:
        k = min((f - 1) // per, len(rects) - 1)
        r = rects[k]
        t = ((f - 1) % per) / max(1, per - 1)
        alpha = math.sin(math.pi * t)  # fade in then out
        pad = 6
        g = int(255 - 155 * alpha)  # white -> gray -> white
        d.rectangle([r["x0"] - pad, r["y0"] - pad, r["x1"] + pad, r["y1"] + pad],
                    outline=(g, g, g), width=2)
    else:
        t = ease((f - COMPARE_END) / (N - 1 - COMPARE_END))
        # draw an arc sweeping around, thickness grows to final
        sweep = 360 * t
        width = 5
        bbox = [cx - R, cy - R, cx + R, cy + R]
        if sweep >= 359.5:
            d.ellipse(bbox, outline=(255, 0, 0), width=width)
        else:
            d.arc(bbox, start=-90, end=-90 + sweep, fill=(255, 0, 0), width=width)
    frames.append(img)

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
os.makedirs(tmp, exist_ok=True)
for i, im in enumerate(frames):
    im.save(os.path.join(tmp, f"{i:04d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "12", OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", OUT)
