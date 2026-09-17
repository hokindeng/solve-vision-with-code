#!/usr/bin/env python3
"""Draw the next shape of the size cycle inside the empty dashed box.

Cycle observed in first_frame.png: small, large, medium, small, large -> next = medium.
Only pixels strictly inside the dashed box are ever modified.
"""
import subprocess, os
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# --- detect shapes (purple) and the dashed box (black) -----------------------
purple = (base[:, :, 0] > 100) & (base[:, :, 2] > 200) & (base[:, :, 1] < 150)
color = tuple(int(v) for v in base[purple][0])
cols = purple.any(axis=0)
segs, s = [], None
for x, v in enumerate(cols):
    if v and s is None:
        s = x
    if not v and s is not None:
        segs.append((s, x - 1)); s = None
shapes = []
for x0, x1 in segs:
    ys = np.nonzero(purple[:, x0:x1 + 1].any(axis=1))[0]
    shapes.append(dict(x0=x0, x1=x1, y0=ys.min(), y1=ys.max(),
                       w=x1 - x0 + 1, cx=(x0 + x1) / 2, cy=(ys.min() + ys.max()) / 2))

dark = base.sum(axis=2) < 200
dys, dxs = np.nonzero(dark)
bx0, bx1, by0, by1 = dxs.min(), dxs.max(), dys.min(), dys.max()
box_cx, box_cy = (bx0 + bx1) / 2, (by0 + by1) / 2

# --- determine the size cycle ---------------------------------------------------
widths = [sh["w"] for sh in shapes]
sizes = sorted(set(widths))
labels = [sizes.index(w) for w in widths]          # 0=smallest ...
period = None
for p in range(1, len(labels)):
    if all(labels[i] == labels[i - p] for i in range(p, len(labels))):
        period = p; break
next_label = labels[len(labels) % period] if period else labels[0]
template = shapes[widths.index(sizes[next_label])]
print(f"sizes (px): {widths} -> labels {labels}, period {period}, next label {next_label} "
      f"(width {sizes[next_label]})")

# exact pixel copy of the template shape, translated into the box
tmpl_mask = purple[template["y0"]:template["y1"] + 1, template["x0"]:template["x1"] + 1]
dx = int(round(box_cx - template["cx"]))
dy = int(round(box_cy - template["cy"]))
tx0, ty0 = template["x0"] + dx, template["y0"] + dy
final_mask = np.zeros((H, W), bool)
final_mask[ty0:ty0 + tmpl_mask.shape[0], tx0:tx0 + tmpl_mask.shape[1]] = tmpl_mask

# the interior of the box (strictly inside the dashed border) is the only editable region
inner = np.zeros((H, W), bool)
inner[by0 + 3:by1 - 2, bx0 + 3:bx1 - 2] = True
assert final_mask[~inner].sum() == 0

r_full = (template["x1"] - template["x0"]) / 2.0  # half-diagonal of the diamond


def ease(t):
    return t * t * (3 - 2 * t)


def frame(i):
    img = base.copy()
    hold_start, grow_end = 12, 50
    if i < hold_start:
        return img
    if i >= grow_end:
        img[final_mask] = color
        return img
    t = ease((i - hold_start + 1) / (grow_end - hold_start))
    r = r_full * t
    pil = Image.new("L", (W, H), 0)
    ImageDraw.Draw(pil).polygon(
        [(box_cx - r, box_cy), (box_cx, box_cy - r), (box_cx + r, box_cy), (box_cx, box_cy + r)],
        fill=255)
    m = (np.array(pil) > 0) & inner
    img[m] = color
    return img


os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
os.makedirs(tmp, exist_ok=True)
for i in range(N_FRAMES):
    Image.fromarray(frame(i)).save(os.path.join(tmp, f"{i:04d}.png"))

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
                "-r", str(FPS), OUT], check=True)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print("wrote", OUT)
