#!/usr/bin/env python3
"""Rearrange shapes: group by type, sort each group small->large, lay out in one row."""
import os, subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 96
HOLD_START, HOLD_END = 6, 10           # frames held still at start / end

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg_color = np.array([235, 235, 235], dtype=np.uint8)
mask = np.any(img != bg_color, axis=2)
lab, n = ndimage.label(mask)

def classify(sub_mask):
    """circle vs triangle from fill ratio of bounding box (circle ~0.785, triangle ~0.5)."""
    ratio = sub_mask.sum() / sub_mask.size
    return "circle" if ratio > 0.65 else "triangle"

shapes = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = (lab[y0:y1, x0:x1] == i)
    shapes.append(dict(
        kind=classify(m), w=x1 - x0, h=y1 - y0, x0=int(x0), y0=int(y0),
        sprite=img[y0:y1, x0:x1].copy(), mask=m, area=int(m.sum()),
        cx=(x0 + x1) / 2.0,
    ))

# Group order: the type whose shapes are on average further left comes first.
kinds = sorted({s["kind"] for s in shapes},
               key=lambda k: np.mean([s["cx"] for s in shapes if s["kind"] == k]))
ordered = []
for k in kinds:
    ordered += sorted([s for s in shapes if s["kind"] == k], key=lambda s: s["area"])

# Target layout: single horizontal row, vertically centred, even gaps.
margin = 40
total_w = sum(s["w"] for s in ordered)
gap = (W - 2 * margin - total_w) / (len(ordered) - 1)
x = margin
for s in ordered:
    s["tx0"] = int(round(x))
    s["ty0"] = int(round(H / 2 - s["h"] / 2))
    x += s["w"] + gap

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)   # smooth in/out

move_frames = N_FRAMES - HOLD_START - HOLD_END
def global_t(f):
    if f < HOLD_START: return 0.0
    if f >= N_FRAMES - HOLD_END: return 1.0
    return (f - HOLD_START) / (move_frames - 1)

def positions(scheme, T):
    """Return (px, py) for each shape in `ordered` at global progress T in [0,1]."""
    out = []
    k = len(ordered)
    for i, s in enumerate(ordered):
        if scheme == "simultaneous":
            t = ease(T)
        elif scheme.startswith("sequential"):     # one shape at a time
            idx = i if scheme == "sequential" else k - 1 - i
            t = ease(np.clip(T * k - idx, 0, 1))
        elif scheme == "staggered":                # overlapping starts
            span = 0.55
            t = ease(np.clip((T - i * (1 - span) / (k - 1)) / span, 0, 1))
        out.append((int(round(s["x0"] + (s["tx0"] - s["x0"]) * t)),
                    int(round(s["y0"] + (s["ty0"] - s["y0"]) * t))))
    return out

def overlap_cost(scheme):
    cost = 0
    for f in range(N_FRAMES):
        cover = np.zeros((H, W), dtype=np.uint8)
        for s, (px, py) in zip(ordered, positions(scheme, global_t(f))):
            cover[py:py + s["h"], px:px + s["w"]] += s["mask"]
        cost += int((cover > 1).sum())
    return cost

schemes = ["simultaneous", "sequential", "sequential_rev", "staggered"]
costs = {sc: overlap_cost(sc) for sc in schemes}
scheme = min(schemes, key=lambda sc: costs[sc])
print("overlap costs:", costs, "-> using", scheme)

def render(f):
    frame = np.empty_like(img); frame[:] = bg_color
    for s, (px, py) in zip(ordered, positions(scheme, global_t(f))):
        region = frame[py:py + s["h"], px:px + s["w"]]
        region[s["mask"]] = s["sprite"][s["mask"]]
    return frame

os.makedirs(OUT_DIR, exist_ok=True)
frames = [img.copy() if global_t(f) == 0.0 else render(f) for f in range(N_FRAMES)]

p = subprocess.Popen([
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT,
], stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
print("wrote", OUT, "order:", [(s["kind"], s["area"]) for s in ordered])
