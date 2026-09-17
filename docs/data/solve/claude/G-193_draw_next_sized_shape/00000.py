#!/usr/bin/env python3
"""Draw the next shape in the size cycle inside the empty box, step by step."""
import os, subprocess, shutil
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 60

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape

# ---- Analyse the scene -------------------------------------------------
a = base.astype(int)
white = (a == 255).all(2)
black = (a == 0).all(2)
shape_mask = ~white & ~black
color = tuple(int(v) for v in base[shape_mask][0])

lab, n = ndimage.label(shape_mask)
shapes = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    shapes.append(dict(idx=i, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                       cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2,
                       w=xs.max() - xs.min() + 1))
shapes.sort(key=lambda s: s["cx"])

# Box (dashed black rectangle)
ys, xs = np.where(black)
bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
box_cx, box_cy = (bx0 + bx1) / 2, (by0 + by1) / 2
# line thickness -> interior
col = black[:, int(box_cx)]
t = int(np.argmax(~col[by0:]))  # first non-black after top edge
ix0, ix1, iy0, iy1 = bx0 + t, bx1 - t, by0 + t, by1 - t

# Size cycle: cluster widths into distinct sizes, find period
widths = [s["w"] for s in shapes]
sizes = sorted(set(widths))
seq = [sizes.index(w) for w in widths]
period = next(p for p in range(1, len(seq) + 1)
              if all(seq[i] == seq[i % p] for i in range(len(seq))))
next_size_idx = seq[len(seq) % period]
template = next(s for s in shapes if sizes.index(s["w"]) == next_size_idx)
print(f"sizes={sizes} seq={seq} period={period} -> next width {sizes[next_size_idx]}")

# Exact copy of template shape, translated to the box centre
tmask = lab == template["idx"]
dx = int(round(box_cx - template["cx"]))
dy = int(round(box_cy - template["cy"]))
final_mask = np.roll(np.roll(tmask, dy, axis=0), dx, axis=1)
r = template["w"] / 2.0
cx, cy = template["cx"] + dx, template["cy"] + dy

interior = np.zeros((H, W), bool)
interior[iy0:iy1 + 1, ix0:ix1 + 1] = True
assert (final_mask & ~interior).sum() == 0


def compose(draw_mask):
    """Return base frame with `color` painted where draw_mask (restricted to box interior)."""
    out = base.copy()
    m = draw_mask & interior
    out[m] = color
    return out


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


frames = []
HOLD0, TRACE, FILL = 10, 18, 22   # 10 + 18 + 22 = 50, then 10 hold
for f in range(N_FRAMES):
    if f < HOLD0:
        frames.append(base.copy())
    elif f < HOLD0 + TRACE:
        # Step 1: trace the outline of the shape that should come next
        p = ease((f - HOLD0 + 1) / TRACE)
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        bb = [cx - r + 0.5, cy - r + 0.5, cx + r - 0.5, cy + r - 0.5]
        d.arc(bb, -90, -90 + 360 * p, fill=255, width=3)
        frames.append(compose(np.array(im) > 0))
    elif f < HOLD0 + TRACE + FILL:
        # Step 2: fill the shape, growing from the centre out to the outline
        p = ease((f - HOLD0 - TRACE + 1) / FILL)
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        bb = [cx - r + 0.5, cy - r + 0.5, cx + r - 0.5, cy + r - 0.5]
        d.arc(bb, 0, 360, fill=255, width=3)
        rr = r * p
        if rr > 0.5:
            d.ellipse([cx - rr + 0.5, cy - rr + 0.5, cx + rr - 0.5, cy + rr - 0.5], fill=255)
        m = (np.array(im) > 0)
        if p >= 1:
            m = final_mask
        frames.append(compose(m))
    else:
        frames.append(compose(final_mask))

# ---- Encode ------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0",
                "-x264-params", "keyint=1", OUT], check=True)
shutil.rmtree(tmp)
print("wrote", OUT, len(frames), "frames")
