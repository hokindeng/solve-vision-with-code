#!/usr/bin/env python3
"""Generate the step-by-step solution video for the size-sequence puzzle.

Frame 0 is exactly first_frame.png.  Annotations are drawn only in empty
(white) regions so every other pixel stays unchanged; the final frames show a
red circle around the correct choice.
"""
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
W = H = 1024
FPS = 16
N_FRAMES = 60
SS = 2  # supersampling factor for anti-aliased overlays

RED = (220, 30, 30)
BLUE = (30, 90, 200)
GREEN = (20, 150, 60)
GRAY = (110, 110, 110)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size * SS)


# --------------------------------------------------------------------------
# Scene analysis
# --------------------------------------------------------------------------
def analyse(base):
    a = np.asarray(base).astype(int)
    dark = a.sum(2) < 60
    lab, n = ndimage.label(dark)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        comps.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                          cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2,
                          w=xs.max() - xs.min() + 1, h=ys.max() - ys.min() + 1,
                          area=len(xs)))
    # divider line between areas
    nonwhite = a.sum(2) < 750
    rows = np.where(nonwhite.sum(1) > 500)[0]
    divider_y = int(rows[0])
    seq = sorted([c for c in comps if c["cy"] < divider_y], key=lambda c: c["cx"])
    # choice boxes (light gray fills)
    boxmask = (abs(a[:, :, 0] - 238) < 3) & (abs(a[:, :, 1] - 238) < 3) & (abs(a[:, :, 2] - 238) < 3)
    lab, n = ndimage.label(boxmask)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) > 5000:
            boxes.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max()))
    boxes.sort(key=lambda b: b["x0"])
    # per-choice shape description
    choices = []
    for b in boxes:
        sub = a[b["y0"]:b["y1"] + 1, b["x0"]:b["x1"] + 1]
        m = np.abs(sub - 238).sum(2) > 30
        ys, xs = np.where(m)
        col = tuple(int(v) for v in np.median(sub[m], axis=0))
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        fill = len(xs) / float(w * h)
        choices.append(dict(box=b, w=w, h=h, color=col, fill=fill,
                            cx=b["x0"] + (xs.min() + xs.max()) / 2,
                            cy=b["y0"] + (ys.min() + ys.max()) / 2))
    # dashed placeholder box
    dash = (np.abs(a - 185) < 12).all(2)
    dash[divider_y - 5:, :] = False
    dash[:, :seq[-1]["x1"] + 10] = False
    ys, xs = np.where(dash)
    dashed = dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max())
    return seq, choices, dashed, divider_y


def solve(seq, choices):
    sizes = [round(c["w"]) for c in seq]
    # sizes are ~28,44,60 (measured 27/43/60 due to raster); use rounded to 4
    sizes = [int(round(s / 4.0) * 4) for s in sizes]
    steps = [b - a for a, b in zip(sizes, sizes[1:])]
    step = int(round(np.mean(steps)))
    target = sizes[-1] + step
    seq_color = (0, 0, 0)
    best, best_err = None, 1e9
    for i, c in enumerate(choices):
        is_square = c["fill"] > 0.9 and abs(c["w"] - c["h"]) <= 2
        same_col = sum(abs(x - y) for x, y in zip(c["color"], seq_color)) < 60
        if is_square and same_col:
            err = abs(c["w"] - target)
            if err < best_err:
                best, best_err = i, err
    return sizes, step, target, best


# --------------------------------------------------------------------------
# Drawing helpers (work on the supersampled overlay)
# --------------------------------------------------------------------------
def S(v):
    return v * SS


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def seg(f, a, b):
    """progress 0..1 of frame f within [a, b)"""
    if f < a:
        return 0.0
    if f >= b:
        return 1.0
    return (f - a) / float(b - a)


def text_center(d, xy, s, fnt, fill):
    x, y = xy
    bb = d.textbbox((0, 0), s, font=fnt)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text((S(x) - w / 2 - bb[0], S(y) - h / 2 - bb[1]), s, font=fnt, fill=fill)


def bracket(d, x0, x1, y, prog, fill, width=2):
    """horizontal measuring bracket from x0 to x1 at height y, growing."""
    if prog <= 0:
        return
    xe = x0 + (x1 - x0) * prog
    d.line([(S(x0), S(y)), (S(xe), S(y))], fill=fill, width=S(width))
    d.line([(S(x0), S(y - 5)), (S(x0), S(y + 5))], fill=fill, width=S(width))
    if prog >= 1:
        d.line([(S(x1), S(y - 5)), (S(x1), S(y + 5))], fill=fill, width=S(width))


def arrow(d, p0, p1, prog, fill, width=2):
    if prog <= 0:
        return
    x0, y0 = p0
    x1, y1 = p1
    xe, ye = x0 + (x1 - x0) * prog, y0 + (y1 - y0) * prog
    d.line([(S(x0), S(y0)), (S(xe), S(ye))], fill=fill, width=S(width))
    if prog >= 1:
        v = np.array([x1 - x0, y1 - y0], float)
        v /= np.linalg.norm(v)
        n = np.array([-v[1], v[0]])
        tip = np.array([x1, y1])
        a1 = tip - v * 8 + n * 5
        a2 = tip - v * 8 - n * 5
        d.polygon([tuple(S(tip)), tuple(S(a1)), tuple(S(a2))], fill=fill)


def cross(d, cx, cy, r, prog, fill, width=3):
    if prog <= 0:
        return
    p1 = min(prog * 2, 1.0)
    p2 = max(0.0, prog * 2 - 1)
    d.line([(S(cx - r), S(cy - r)), (S(cx - r + 2 * r * p1), S(cy - r + 2 * r * p1))],
           fill=fill, width=S(width))
    if p2 > 0:
        d.line([(S(cx + r), S(cy - r)), (S(cx + r - 2 * r * p2), S(cy - r + 2 * r * p2))],
               fill=fill, width=S(width))


def check(d, cx, cy, r, prog, fill, width=3):
    if prog <= 0:
        return
    pts = [(cx - r, cy), (cx - r * 0.3, cy + r * 0.7), (cx + r, cy - r * 0.8)]
    total = 2
    cur = prog * total
    path = [pts[0]]
    for i in range(total):
        if cur >= i + 1:
            path.append(pts[i + 1])
        elif cur > i:
            t = cur - i
            path.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
                         pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t))
            break
    if len(path) > 1:
        d.line([(S(x), S(y)) for x, y in path], fill=fill, width=S(width), joint="curve")


def circle_arc(d, cx, cy, r, prog, fill, width=5):
    if prog <= 0:
        return
    bbox = [S(cx - r), S(cy - r), S(cx + r), S(cy + r)]
    start = -90
    end = start + 360 * prog
    if prog >= 1:
        d.ellipse(bbox, outline=fill, width=S(width))
    else:
        d.arc(bbox, start=start, end=end, fill=fill, width=S(width))


# --------------------------------------------------------------------------
# Frame rendering
# --------------------------------------------------------------------------
def render(f, base, seq, choices, dashed, divider_y, sizes, step, target, answer):
    if f == 0:
        return base.copy()
    ov = Image.new("RGBA", (S(W), S(H)), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f_small = font(18)
    f_med = font(22, bold=True)
    f_big = font(26, bold=True)

    # ---- Step 1 (frames 2..20): measure each sequence shape -------------
    label_y = seq[0]["y0"] - 46  # above the largest shape top (y0 ~308) -> ~262
    top_min = min(c["y0"] for c in seq)
    label_y = top_min - 30
    for i, c in enumerate(seq):
        p = ease(seg(f, 2 + i * 4, 6 + i * 4))
        bracket(d, c["x0"], c["x1"] + 1, label_y, p, BLUE)
        if p >= 1:
            text_center(d, (c["cx"], label_y - 18), str(sizes[i]), f_med, BLUE)

    # ---- Step 2 (frames 14..28): step arrows "+16" between shapes ---------
    arrow_y = max(c["y1"] for c in seq) + 40
    for i in range(len(seq) - 1):
        p = ease(seg(f, 14 + i * 4, 19 + i * 4))
        a, b = seq[i], seq[i + 1]
        arrow(d, (a["cx"] + 12, arrow_y), (b["cx"] - 12, arrow_y), p, GREEN)
        if p >= 1:
            text_center(d, ((a["cx"] + b["cx"]) / 2, arrow_y + 20), "+%d" % step, f_med, GREEN)
    # arrow into dashed box + predicted size
    p = ease(seg(f, 22, 27))
    last = seq[-1]
    dcx = (dashed["x0"] + dashed["x1"]) / 2
    dcy = (dashed["y0"] + dashed["y1"]) / 2
    arrow(d, (last["cx"] + 12, arrow_y), (dcx - 12, arrow_y), p, GREEN)
    if p >= 1:
        text_center(d, ((last["cx"] + dcx) / 2, arrow_y + 20), "+%d" % step, f_med, GREEN)
    p = ease(seg(f, 27, 31))
    if p > 0:
        # ghost prediction square growing inside the dashed box
        half = target / 2 * p
        d.rectangle([S(dcx - half), S(dcy - half), S(dcx + half), S(dcy + half)],
                    outline=RED, width=S(2))
        if p >= 1:
            text_center(d, (dcx, label_y - 18), "%d + %d = %d" % (sizes[-1], step, target), f_med, RED)
    # step summary text under the sequence
    p = seg(f, 31, 32)
    if p >= 1:
        text_center(d, (W / 2, arrow_y + 70),
                    "constant step = %d  ->  next square must be %d px, same shape & color" % (step, target),
                    f_small, GRAY)

    # ---- Step 3 (frames 33..49): check each choice ------------------------
    for i, c in enumerate(choices):
        b = c["box"]
        p = ease(seg(f, 33 + i * 4, 37 + i * 4))
        if p <= 0:
            continue
        mx, my = b["x1"] - 22, b["y0"] + 22
        if i == answer:
            check(d, mx, my, 11, p, GREEN, 4)
        else:
            cross(d, mx, my, 10, p, RED, 4)
        if p >= 1:
            if i == answer:
                lbl = "%d px" % c["w"]
                col = GREEN
            elif c["fill"] < 0.9:
                lbl = "shape"
                col = RED
            elif sum(c["color"]) > 100:
                lbl = "color"
                col = RED
            else:
                lbl = "%d px" % c["w"]
                col = RED
            text_center(d, ((b["x0"] + b["x1"]) / 2, b["y1"] + 22), lbl, f_small, col)

    # ---- Step 4 (frames 49..58): red circle around the answer ----------------
    p = ease(seg(f, 49, 58))
    ans = choices[answer]
    b = ans["box"]
    r = min(b["x1"] - b["x0"], b["y1"] - b["y0"]) / 2 - 4
    circle_arc(d, (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, r, p, RED, 6)

    ov = ov.resize((W, H), Image.LANCZOS)
    out = base.convert("RGBA")
    out.alpha_composite(ov)
    return out.convert("RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    seq, choices, dashed, divider_y = analyse(base)
    sizes, step, target, answer = solve(seq, choices)
    print("sizes", sizes, "step", step, "target", target, "answer option", answer + 1)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        fr = render(f, base, seq, choices, dashed, divider_y, sizes, step, target, answer)
        proc.stdin.write(np.asarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
