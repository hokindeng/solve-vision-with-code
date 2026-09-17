#!/usr/bin/env python3
"""Generate the step-by-step solution video for the large/small alternation task."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import imageio.v2 as imageio
import os, subprocess, math

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16

base = Image.open(SRC).convert("RGB")
W, H = base.size
arr = np.array(base).astype(int)

# ---------------- scene analysis ----------------
def components(mask, min_area=50):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < min_area:
            continue
        out.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                        area=len(xs), cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2))
    return out

# separator line: a long dark-ish horizontal row
gray = arr.mean(axis=2)
row_dark = (gray < 235).sum(axis=1)
sep_y = int(np.argmax(row_dark))

# colored (saturated) pixels = shapes
sat = arr.max(axis=2) - arr.min(axis=2)
shape_mask = sat > 40

top_shapes = sorted([c for c in components(shape_mask) if c['cy'] < sep_y], key=lambda c: c['cx'])
bot_shapes = sorted([c for c in components(shape_mask) if c['cy'] > sep_y], key=lambda c: c['cx'])

# option cards: light gray (245) regions below the separator
card_mask = (np.abs(arr - 245).max(axis=2) <= 2)
card_mask[:sep_y] = False
cards = sorted([c for c in components(card_mask, 5000)], key=lambda c: c['cx'])

# placeholder "?" box: gray dashed box in top area
dash_mask = (np.abs(arr - 170).max(axis=2) <= 3)
dash_mask[sep_y - 5:] = False
ys, xs = np.where(dash_mask)
qbox = dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max())

def size_of(c):
    return max(c['x1'] - c['x0'], c['y1'] - c['y0'])

def color_of(c):
    y, x = int(c['cy']), int(c['cx'])
    return tuple(arr[y, x])

def shape_kind(c):
    w = c['x1'] - c['x0'] + 1; h = c['y1'] - c['y0'] + 1
    fill = c['area'] / (w * h)
    if fill > 0.9: return 'square'
    if fill > 0.7: return 'circle'
    return 'triangle'

sizes = [size_of(c) for c in top_shapes]
thr = (min(sizes) + max(sizes)) / 2
labels = ['LARGE' if s > thr else 'SMALL' for s in sizes]
next_label = 'SMALL' if labels[-1] == 'LARGE' else 'LARGE'
seq_color = color_of(top_shapes[0]); seq_kind = shape_kind(top_shapes[0])

def matches(c):
    col = color_of(c)
    if max(abs(a - b) for a, b in zip(col, seq_color)) > 30: return False
    if shape_kind(c) != seq_kind: return False
    lab = 'LARGE' if size_of(c) > thr else 'SMALL'
    return lab == next_label

correct = [i for i, c in enumerate(bot_shapes) if matches(c)]
assert len(correct) == 1, correct
correct = correct[0]
print("sequence:", labels, "-> next:", next_label, "| correct option:", correct + 1)

# ---------------- drawing helpers ----------------
def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
F_TAG = font(22)

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

def blend(img, overlay, alpha):
    if alpha <= 0: return img
    if alpha >= 1: return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    a = overlay.split()[3].point(lambda v: int(v * alpha))
    ov = overlay.copy(); ov.putalpha(a)
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

BLUE = (30, 110, 220, 255)
RED = (220, 30, 30, 255)

def tag_layer(c, text, color=BLUE, above=True):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    pad = 10
    d.rounded_rectangle([c['x0'] - pad, c['y0'] - pad, c['x1'] + pad, c['y1'] + pad],
                        radius=6, outline=color, width=3)
    tw = d.textlength(text, font=F_TAG)
    tx = c['cx'] - tw / 2
    ty = c['y0'] - pad - 34 if above else c['y1'] + pad + 8
    d.text((tx, ty), text, font=F_TAG, fill=color)
    return ov

def circle_layer(card, frac):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    cx, cy = card['cx'], card['cy']
    r = min(card['x1'] - card['x0'], card['y1'] - card['y0']) / 2 - 6
    bbox = [cx - r, cy - r, cx + r, cy + r]
    end = -90 + 360 * frac
    if frac >= 1: d.ellipse(bbox, outline=RED, width=7)
    elif frac > 0: d.arc(bbox, -90, end, fill=RED, width=7)
    return ov

# ---------------- timeline ----------------
# frames 0: original; 2-26: annotate sequence items + next-size deduction;
# 26-40: scan the options; 40-54: draw red circle; 54-59: fade annotations, keep circle.
seq_items = [(top_shapes[i], labels[i]) for i in range(len(top_shapes))]
qc = dict(x0=qbox['x0'], x1=qbox['x1'], y0=qbox['y0'], y1=qbox['y1'],
          cx=(qbox['x0'] + qbox['x1']) / 2, cy=(qbox['y0'] + qbox['y1']) / 2)
seq_starts = [2 + 6 * i for i in range(len(seq_items))]          # 2, 8, 14
q_start = seq_starts[-1] + 6                                       # 20
scan_start, scan_len = 26, 3.5                                     # 26..40
circ_start, circ_end = 40, 54
fade_start, fade_end = 54, 59

frames = []
for f in range(N_FRAMES):
    img = base.copy()
    ann_alpha = 1.0
    if f >= fade_start:
        ann_alpha = 1.0 - ease((f - fade_start) / (fade_end - fade_start))
    # sequence tags
    for (c, lab), s in zip(seq_items, seq_starts):
        if f >= s:
            a = ease((f - s) / 3) * ann_alpha
            img = blend(img, tag_layer(c, lab), a)
    if f >= q_start:
        a = ease((f - q_start) / 3) * ann_alpha
        img = blend(img, tag_layer(qc, next_label + " ?"), a)
    # option scan
    if scan_start <= f < circ_start:
        k = min(int((f - scan_start) / scan_len), len(cards) - 1)
        card = cards[k]
        col = (40, 170, 90, 255) if k == correct else (150, 150, 150, 255)
        text = "match" if k == correct else "no"
        img = blend(img, tag_layer(card, text, col, above=False), ann_alpha)
    elif f >= circ_start and f < fade_end:
        pass
    # red circle
    if f >= circ_start:
        frac = ease((f - circ_start) / (circ_end - circ_start))
        img = blend(img, circle_layer(cards[correct], frac), 1.0)
    frames.append(np.array(img))

frames[0] = np.array(base)  # exact first frame

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames.npy")
w = imageio.get_writer(OUT, fps=FPS, codec="libx264", pixelformat="yuv420p",
                       ffmpeg_params=["-crf", "12", "-preset", "slow"], macro_block_size=1)
for fr in frames: w.append_data(fr)
w.close()
print("wrote", OUT, len(frames), "frames")
