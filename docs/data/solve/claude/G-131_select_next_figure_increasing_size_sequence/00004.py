#!/usr/bin/env python3
"""Solve the 'next figure in an increasing-size sequence' puzzle and render the
solution process as a short video.

Pipeline
  1. Analyse /app/first_frame.png: find the separator line, the sequence shapes
     (top area), the placeholder box and the four choice cells (bottom area).
  2. Measure each shape (colour, vertex count => shape kind, bounding size).
  3. Derive the constant size step, predict the next size and pick the option
     that matches shape, colour and predicted size.
  4. Render 60 frames at 16 fps: measure the sequence, show the step, preview
     the expected figure in the placeholder, check each option, then circle the
     correct one in red.  Every pixel not touched by these overlays stays
     identical to the first frame.
"""
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 60

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

RED = (220, 30, 30)
INK = (60, 60, 70)          # annotation text colour
ACCENT = (40, 110, 220)     # highlight colour for measuring
OK = (30, 160, 70)
BAD = (200, 60, 60)


# --------------------------------------------------------------------------- #
# Scene analysis
# --------------------------------------------------------------------------- #
def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def find_separator(img):
    """Row of the long horizontal grey line splitting the two areas: the first
    row whose non-white pixels form one contiguous run spanning most of the width."""
    nonwhite = np.abs(img.astype(int) - 255).sum(-1) > 0
    for y in range(H):
        xs = np.where(nonwhite[y])[0]
        if len(xs) > 0.6 * W and (xs[-1] - xs[0] + 1) == len(xs):
            return int(y)
    return H // 2


def classify_polygon(mask):
    """Return (kind, n_vertices) for a filled shape mask."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    n = len(approx)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    fill = area / float(w * h)
    if n >= 8 and fill > 0.7:
        return "circle", n
    if n == 3:
        return "triangle", n
    if n == 4:
        # diamond (rotated square) has ~0.5 bbox fill, square ~1.0
        return ("diamond" if fill < 0.7 else "square"), n
    if n == 5:
        return "pentagon", n
    if n == 6:
        return "hexagon", n
    return "star" if fill < 0.5 else f"poly{n}", n


def find_shapes(img, y0, y1, ignore):
    """Coloured (saturated) connected components inside rows y0..y1."""
    sub = img[y0:y1]
    mx = sub.max(-1).astype(int)
    mn = sub.min(-1).astype(int)
    sat = (mx - mn) > 40  # exclude white / greys
    n, lab, stats, cent = cv2.connectedComponentsWithStats(sat.astype(np.uint8), 8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 80:
            continue
        m = lab == i
        col = tuple(int(v) for v in np.median(sub[m], axis=0))
        kind, nv = classify_polygon(m)
        shapes.append(dict(x=int(x), y=int(y) + y0, w=int(w), h=int(h),
                           size=int(max(w, h)), cx=float(cent[i][0]),
                           cy=float(cent[i][1]) + y0, color=col, kind=kind))
    shapes.sort(key=lambda s: s["cx"])
    return shapes


def find_boxes(img, y0, y1, color, min_w=60):
    """Bounding boxes of components made of an exact grey colour."""
    sub = img[y0:y1]
    m = np.all(sub == np.array(color, np.uint8), -1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w >= min_w and h >= min_w:
            boxes.append((int(x), int(y) + y0, int(w), int(h)))
    boxes.sort()
    return boxes


def find_placeholder(img, y0, y1):
    """Dashed square in the sequence area: union of small grey dash components."""
    sub = img[y0:y1]
    g = (np.abs(sub.astype(int) - sub[..., :1].astype(int)).sum(-1) < 6) & \
        (sub[..., 0] < 235) & (sub[..., 0] > 120)
    ys, xs = np.where(g)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()) + y0,
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def analyse(img):
    sep = find_separator(img)
    seq = find_shapes(img, 0, sep - 2, None)
    placeholder = find_placeholder(img, 0, sep - 2)
    cells = find_boxes(img, sep + 2, H, (200, 200, 200))
    choices = find_shapes(img, sep + 2, H, None)
    # attach each choice shape to its cell
    for c in choices:
        for (x, y, w, h) in cells:
            if x <= c["cx"] <= x + w and y <= c["cy"] <= y + h:
                c["cell"] = (x, y, w, h)
    sizes = [s["size"] for s in seq]
    steps = [b - a for a, b in zip(sizes, sizes[1:])]
    step = int(round(float(np.median(steps)))) if steps else 0
    predicted = sizes[-1] + step
    ref = seq[-1]

    def color_close(a, b):
        return max(abs(int(u) - int(v)) for u, v in zip(a, b)) < 40

    verdicts = []
    best, best_err = None, 1e9
    for i, c in enumerate(choices):
        why = []
        if c["kind"] != ref["kind"]:
            why.append("shape")
        if not color_close(c["color"], ref["color"]):
            why.append("color")
        err = abs(c["size"] - predicted)
        if err > max(4, 0.25 * step):
            why.append("size")
        verdicts.append(why)
        if not why and err < best_err:
            best, best_err = i, err
    if best is None:  # fallback: closest size among same shape/colour
        best = int(np.argmin([abs(c["size"] - predicted) for c in choices]))
    return dict(sep=sep, seq=seq, placeholder=placeholder, choices=choices,
                sizes=sizes, step=step, predicted=predicted, best=best,
                verdicts=verdicts)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def seg(f, a, b):
    """Progress 0..1 of frame f within [a, b)."""
    if b <= a:
        return 1.0 if f >= a else 0.0
    return min(max((f - a) / float(b - a), 0.0), 1.0)


def blend(base, overlay, alpha):
    """Alpha-composite an RGBA overlay onto RGB base, scaled by alpha."""
    if alpha <= 0:
        return base
    ov = overlay.copy()
    if alpha < 1:
        a = ov.getchannel("A").point(lambda v: int(v * alpha))
        ov.putalpha(a)
    out = base.convert("RGBA")
    out.alpha_composite(ov)
    return out.convert("RGB")


def text_center(d, xy, s, fnt, fill):
    x, y = xy
    bb = d.textbbox((0, 0), s, font=fnt)
    d.text((x - (bb[2] - bb[0]) / 2 - bb[0], y - (bb[3] - bb[1]) / 2 - bb[1]),
           s, font=fnt, fill=fill)


def dashed_polygon(d, pts, fill, width=3, dash=9, gap=6):
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if L == 0:
            continue
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            d.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)],
                   fill=fill, width=width)
            s += dash + gap


def shape_points(kind, cx, cy, size):
    r = size / 2.0
    if kind == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if kind == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if kind == "triangle":
        return [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    n = {"pentagon": 5, "hexagon": 6}.get(kind, 24)
    return [(cx + r * np.cos(-np.pi / 2 + 2 * np.pi * k / n),
             cy + r * np.sin(-np.pi / 2 + 2 * np.pi * k / n)) for k in range(n)]


def arc_polyline(cx, cy, r, frac, start=-90):
    pts = []
    n = max(2, int(200 * frac))
    for k in range(n + 1):
        a = np.deg2rad(start + 360 * frac * k / n)
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return pts


# --------------------------------------------------------------------------- #
# Frame rendering
# --------------------------------------------------------------------------- #
# Timeline (frames)
T_MEASURE = (1, 19)     # size labels under each sequence shape, one at a time
T_STEP = (14, 28)       # "+step" arrows between shapes, rule text
T_PREDICT = (24, 36)    # ghost of expected figure grows in the placeholder
T_CHECK = (34, 50)      # check each option left to right
T_CIRCLE = (48, 58)     # red circle sweeps around the correct option


def render_frame(base, info, f):
    if f == 0:
        return base.copy()
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    seq, sizes, step = info["seq"], info["sizes"], info["step"]
    pred = info["predicted"]
    ph = info["placeholder"]
    f_lab = font(22)
    f_small = font(18)
    f_big = font(26, bold=True)
    label_y = max(s["y"] + s["h"] for s in seq) + 34

    # ---- step 1: measure each sequence shape ------------------------------ #
    n = len(seq)
    a0, a1 = T_MEASURE
    for i, s in enumerate(seq):
        t = seg(f, a0 + i * (a1 - a0) / n, a0 + (i + 1) * (a1 - a0) / n)
        if t <= 0:
            continue
        al = int(255 * ease(min(1, t * 1.5)))
        # thin measuring bracket (bounding box outline) and size label
        pad = 6
        d.rectangle([s["x"] - pad, s["y"] - pad, s["x"] + s["w"] + pad,
                     s["y"] + s["h"] + pad], outline=ACCENT + (al,), width=2)
        text_center(d, (s["cx"], label_y), f"{s['size']}", f_lab, INK + (al,))

    # ---- step 2: the constant step ---------------------------------------- #
    b0, b1 = T_STEP
    for i in range(n - 1):
        t = seg(f, b0 + i * (b1 - b0) / (n), b0 + (i + 1) * (b1 - b0) / n)
        if t <= 0:
            continue
        al = int(255 * ease(t))
        x0 = seq[i]["cx"] + 22
        x1 = seq[i + 1]["cx"] - 22
        xm = (seq[i]["cx"] + seq[i + 1]["cx"]) / 2
        d.line([(x0, label_y), (x1, label_y)], fill=ACCENT + (al,), width=2)
        d.polygon([(x1, label_y), (x1 - 8, label_y - 5), (x1 - 8, label_y + 5)],
                  fill=ACCENT + (al,))
        text_center(d, (xm, label_y + 22), f"+{sizes[i + 1] - sizes[i]}", f_small,
                    ACCENT + (al,))
    t = seg(f, b1 - 4, b1 + 2)
    if t > 0:
        al = int(255 * ease(t))
        text_center(d, (W / 2, label_y + 70),
                    f"Same shape, same color, size step = +{step}", f_big, INK + (al,))

    # ---- step 3: predicted figure in the placeholder ----------------------- #
    c0, c1 = T_PREDICT
    t = seg(f, c0, c1)
    if t > 0 and ph is not None:
        px, py = ph[0] + ph[2] / 2.0, ph[1] + ph[3] / 2.0
        cur = pred * ease(t)
        col = seq[-1]["color"]
        pts = shape_points(seq[-1]["kind"], px, py, cur)
        if cur > 2:
            dashed_polygon(d, pts, col + (230,), width=3)
            d.polygon(pts, fill=col + (60,))
        # arrow from last shape to placeholder
        al = int(255 * ease(t))
        d.line([(seq[-1]["cx"] + 22, label_y), (px - 34, label_y)],
               fill=ACCENT + (al,), width=2)
        d.polygon([(px - 34, label_y), (px - 42, label_y - 5), (px - 42, label_y + 5)],
                  fill=ACCENT + (al,))
        text_center(d, ((seq[-1]["cx"] + px) / 2, label_y + 22), f"+{step}", f_small,
                    ACCENT + (al,))
        if t > 0.5:
            al2 = int(255 * ease((t - 0.5) * 2))
            text_center(d, (px, label_y), f"{pred}?", f_lab, INK + (al2,))

    # ---- step 4: check the options ---------------------------------------- #
    e0, e1 = T_CHECK
    m = len(info["choices"])
    for i, c in enumerate(info["choices"]):
        t = seg(f, e0 + i * (e1 - e0) / m, e0 + (i + 1) * (e1 - e0) / m)
        if t <= 0:
            continue
        al = int(255 * ease(min(1, t * 1.6)))
        cell = c.get("cell")
        why = info["verdicts"][i]
        if cell is None:
            continue
        x, y, w, h = cell
        ok = len(why) == 0
        colr = (OK if ok else BAD) + (al,)
        d.rectangle([x - 3, y - 3, x + w + 3, y + h + 3], outline=colr, width=3)
        ty = y + h + 28
        if ok:
            msg = f"✓  {c['size']} = {sizes[-1]} + {step}"
        else:
            msg = "✗  " + " & ".join(why) + " differ" if len(why) > 1 else \
                  f"✗  {why[0]} differs"
            if why == ["size"]:
                msg = f"✗  size {c['size']} ≠ {pred}"
        text_center(d, (x + w / 2, ty), msg, f_small, colr)

    # ---- step 5: red circle around the correct option ---------------------- #
    g0, g1 = T_CIRCLE
    t = seg(f, g0, g1)
    if t > 0:
        c = info["choices"][info["best"]]
        x, y, w, h = c["cell"]
        cx, cy = x + w / 2.0, y + h / 2.0
        r = 0.5 * max(w, h) * 0.62 + 6
        frac = ease(t)
        pts = arc_polyline(cx, cy, r, frac)
        if len(pts) >= 2:
            d.line(pts, fill=RED + (255,), width=7, joint="curve")
            if frac >= 0.999:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=RED + (255,), width=7)

    return blend(base, ov, 1.0)


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    img = np.array(base)
    info = analyse(img)
    print("sequence sizes:", info["sizes"], "step:", info["step"],
          "predicted:", info["predicted"])
    for i, c in enumerate(info["choices"]):
        print(f"  option {i}: {c['kind']} {c['color']} size {c['size']} ->",
              "OK" if not info["verdicts"][i] else "reject: " + ", ".join(info["verdicts"][i]))
    print("answer: option", info["best"])

    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(N_FRAMES):
        fr = render_frame(base, info, f)
        fr.save(os.path.join(frames_dir, f"{f:04d}.png"))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(frames_dir, "%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
           "-preset", "slow", "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
