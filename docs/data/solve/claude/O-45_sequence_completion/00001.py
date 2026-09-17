"""Generate the color_cycle sequence-completion video.

Pattern in first_frame.png: teal, pink, navy, bisque, teal, ?  -> cycle of 4,
so the missing element is a pink circle. The question mark fades out and the
pink circle grows into place at the 6th slot; nothing else changes.
"""
import numpy as np
from PIL import Image, ImageDraw
import subprocess, os, shutil, tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT = os.path.join(ROOT, "output", "video.mp4")
FPS, N = 16, 25

base = Image.open(FIRST).convert("RGB")
W, H = base.size
arr = np.array(base)
bg = np.array([255, 255, 255], dtype=np.uint8)

# --- detect the existing circles (column runs of non-background pixels) ---
mask = (arr != bg).any(2)
cols = mask.any(0)
runs, s = [], None
for x in range(W + 1):
    on = x < W and cols[x]
    if on and s is None:
        s = x
    elif not on and s is not None:
        runs.append((s, x - 1)); s = None
circles = runs[:-1]           # last run is the question mark
qx0, qx1 = runs[-1]
qy = np.where(mask[:, qx0:qx1 + 1].any(1))[0]
qy0, qy1 = qy.min(), qy.max()

# geometry of the first circle (all circles share it)
x0, x1 = circles[0]
ys = np.where(mask[:, x0:x1 + 1].any(1))[0]
y0, y1 = ys.min(), ys.max()
cy = (y0 + y1) / 2.0
cx_first = (x0 + x1) / 2.0
spacing = (circles[1][0] + circles[1][1]) / 2.0 - cx_first
cx_new = cx_first + spacing * len(circles)
# outline width: count black pixels inward from the left edge on the centre row
row = arr[int(round(cy))]
ow = 0
while tuple(row[x0 + ow]) == (0, 0, 0):
    ow += 1

# fill colours of the circles in order (centre pixel)
fills = [tuple(int(v) for v in arr[int(round(cy)), (a + b) // 2]) for a, b in circles]
# find the shortest cycle length that explains the sequence
period = next(p for p in range(1, len(fills) + 1)
              if all(fills[i] == fills[i % p] for i in range(len(fills))))
answer = fills[len(circles) % period]
print("fills:", fills, "period:", period, "answer:", answer)

half_w = (x1 - x0) / 2.0
half_h = (y1 - y0) / 2.0
bx0, bx1 = x0 + spacing * len(circles), x1 + spacing * len(circles)
by0, by1 = y0, y1

def ease(t):
    return 1 - (1 - t) ** 3

def frame(i):
    t = i / (N - 1)
    img = arr.copy()
    # phase 1: question mark fades to background over the first ~40%
    f = min(1.0, t / 0.4)
    reg = img[qy0:qy1 + 1, qx0:qx1 + 1].astype(np.float32)
    img[qy0:qy1 + 1, qx0:qx1 + 1] = np.clip(reg * (1 - f) + 255.0 * f, 0, 255).astype(np.uint8)
    # phase 2: circle grows in from ~25% to 100%
    g = max(0.0, (t - 0.25) / 0.75)
    if g > 0:
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        if i == N - 1:
            box = [int(bx0), int(by0), int(bx1), int(by1)]
            d.ellipse(box, fill=answer, outline=(0, 0, 0), width=ow)
        else:
            s = ease(g)
            hw, hh = half_w * s, half_h * s
            box = [cx_new - hw, cy - hh, cx_new + hw, cy + hh]
            w = max(1, int(round(ow * s)))
            d.ellipse(box, fill=answer, outline=(0, 0, 0), width=w)
        img = np.array(pil)
    return img

tmp = tempfile.mkdtemp()
for i in range(N):
    Image.fromarray(frame(i)).save(os.path.join(tmp, f"f{i:03d}.png"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "f%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
                "-r", str(FPS), OUT], check=True)
shutil.rmtree(tmp)
print("wrote", OUT)
