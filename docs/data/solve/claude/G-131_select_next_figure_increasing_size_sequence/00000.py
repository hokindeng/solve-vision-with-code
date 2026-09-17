#!/usr/bin/env python3
"""Generate /app/output/video.mp4: step-by-step solution of the
"select the next figure in an increasing-size sequence" puzzle.

Frame 0 is exactly first_frame.png. Annotations are then added over time:
  1. measure each sequence shape (diameter labels + brackets)
  2. show the constant step (+18) and the predicted next size (75)
  3. check each of the 4 options against shape / colour / size
  4. draw a red circle around the correct option (option 2)
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 60

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

GRAY = (90, 90, 90)
BLUE = (30, 100, 200)
GREEN = (20, 150, 70)
RED = (220, 30, 30)
ORANGE = (255, 140, 0)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)


# --------------------------------------------------------------------------
# Scene analysis (measure the shapes directly from the frame)
# --------------------------------------------------------------------------
def components(mask, min_area=20):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < min_area:
            continue
        out.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                        cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2,
                        w=xs.max() - xs.min() + 1, h=ys.max() - ys.min() + 1,
                        area=len(ys)))
    return out


def analyse(img):
    a = np.asarray(img).astype(int)
    # coloured (non-grey) pixels
    sat = (a.max(axis=2) - a.min(axis=2)) > 60
    comps = components(sat)
    seq = sorted([c for c in comps if c["cy"] < 700], key=lambda c: c["cx"])
    choices = sorted([c for c in comps if c["cy"] > 700], key=lambda c: c["cx"])

    def colour(c):
        return tuple(a[int(c["cy"]), int(c["cx"])])

    def fill_ratio(c):
        return c["area"] / (c["w"] * c["h"])

    seq_col = colour(seq[0])
    sizes = [c["w"] for c in seq]
    step = int(round(np.mean(np.diff(sizes))))
    target = sizes[-1] + step

    # dashed placeholder box (grey pixels in the top area, right of the last shape)
    grey = (np.abs(a[:, :, 0] - a[:, :, 1]) < 8) & (np.abs(a[:, :, 1] - a[:, :, 2]) < 8) \
        & (a[:, :, 0] < 215) & (a[:, :, 0] > 120)
    grey[700:, :] = False
    ys, xs = np.where(grey)
    box = (xs.min(), ys.min(), xs.max(), ys.max())

    # choice panels: light grey (238) runs along a row inside the panels
    row = a[760, :, 0]
    xs = np.where(row == 238)[0]
    runs, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x != p + 1:
            runs.append((s, p))
            s = x
        p = x
    runs.append((s, p))
    col = a[:, runs[0][0] + 5, 0]
    ys = np.where(col == 238)[0]
    panels = [(r[0] - 1, ys.min() - 1, r[1] + 1, ys.max() + 1) for r in runs]

    # evaluate each option
    verdicts = []
    for c in choices:
        same_col = np.abs(np.array(colour(c)) - np.array(seq_col)).max() < 40
        is_circle = abs(fill_ratio(c) - math.pi / 4) < 0.06
        size_ok = abs(c["w"] - target) <= 4
        if not is_circle:
            verdicts.append(("shape", False))
        elif not same_col:
            verdicts.append(("color", False))
        elif not size_ok:
            verdicts.append(("size", False))
        else:
            verdicts.append(("ok", True))
    answer = [i for i, v in enumerate(verdicts) if v[1]][0]
    return dict(seq=seq, sizes=sizes, step=step, target=target, box=box,
                choices=choices, panels=panels, verdicts=verdicts, answer=answer)


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def with_alpha(rgb, alpha):
    return rgb + (int(round(255 * alpha)),)


def text_center(d, xy, s, f, fill):
    bbox = d.textbbox((0, 0), s, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((xy[0] - w / 2 - bbox[0], xy[1] - h / 2 - bbox[1]), s, font=f, fill=fill)


def draw_bracket(d, c, label, alpha, col=BLUE):
    """Horizontal measurement bracket under a sequence shape."""
    x0, x1 = c["x0"], c["x1"]
    y = c["y1"] + 22
    fc = with_alpha(col, alpha)
    d.line([(x0, y), (x1, y)], fill=fc, width=2)
    d.line([(x0, y - 6), (x0, y + 6)], fill=fc, width=2)
    d.line([(x1, y - 6), (x1, y + 6)], fill=fc, width=2)
    text_center(d, ((x0 + x1) / 2, y + 22), label, font(22, True), fc)


def draw_arrow(d, p0, p1, fill, width=3, head=10):
    d.line([p0, p1], fill=fill, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (-1, 1):
        a = ang + math.pi + s * math.radians(28)
        d.line([p1, (p1[0] + head * math.cos(a), p1[1] + head * math.sin(a))],
               fill=fill, width=width)


def dashed_ellipse(d, bbox, fill, width=2, dash=8, gap=6):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    circ = math.pi * (rx + ry)
    n = max(8, int(circ / (dash + gap)))
    for i in range(n):
        a0 = 360 * i / n
        a1 = a0 + 360 * dash / (dash + gap) / n
        d.arc(bbox, a0, a1, fill=fill, width=width)


def render_frame(base, S, k):
    """Return frame k (0..N_FRAMES-1) as an RGB PIL image."""
    if k == 0:
        return base.copy()
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    seq, sizes, step, target = S["seq"], S["sizes"], S["step"], S["target"]
    top_y = 150   # y of the step-caption line in the empty top area

    # ---- Step 1: measure each sequence shape (frames 1..18) ----
    for i, c in enumerate(seq):
        t0 = 1 + 5 * i
        a = ease((k - t0) / 4)
        if a > 0:
            draw_bracket(d, c, f"{sizes[i]}", a)
    a = ease((k - 1) / 4)
    if a > 0:
        text_center(d, (W / 2, top_y), "Step 1: measure the sizes of the sequence shapes",
                    font(24), with_alpha(GRAY, a))

    # ---- Step 2: differences (+18) and predicted next size (frames 16..30) ----
    for i in range(len(seq) - 1):
        t0 = 16 + 4 * i
        a = ease((k - t0) / 4)
        if a > 0:
            c0, c1 = seq[i], seq[i + 1]
            y = c0["y0"] - 36
            p0 = (c0["cx"] + 12, y)
            p1 = (c1["cx"] - 12, y)
            fc = with_alpha(GREEN, a)
            draw_arrow(d, p0, p1, fc)
            text_center(d, ((p0[0] + p1[0]) / 2, y - 18), f"+{sizes[i + 1] - sizes[i]}",
                        font(22, True), fc)
    a = ease((k - 16) / 4)
    if a > 0:
        text_center(d, (W / 2, top_y + 34), f"Step 2: constant size step = +{step}",
                    font(24), with_alpha(GRAY, a))
    # predicted next shape inside the dashed box
    a = ease((k - 24) / 5)
    if a > 0:
        x0, y0, x1, y1 = S["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = target / 2
        fc = with_alpha(ORANGE, a * 0.9)
        dashed_ellipse(d, (cx - r, cy - r, cx + r, cy + r), fc, width=3)
        c_last = seq[-1]
        y = c_last["y0"] - 36
        fg = with_alpha(GREEN, a)
        draw_arrow(d, (c_last["cx"] + 12, y), (cx - 12, y), fg)
        text_center(d, ((c_last["cx"] + cx) / 2, y - 18), f"+{step}", font(22, True), fg)
        text_center(d, (cx, y1 + 44), f"next = {target}", font(22, True),
                    with_alpha(BLUE, a))
        text_center(d, (W / 2, top_y + 68),
                    f"Step 3: the missing shape is an orange circle of size {target}",
                    font(24), with_alpha(GRAY, a))

    # ---- Step 4: check each option (frames 31..46) ----
    labels = {"shape": "shape ✗", "color": "color ✗", "size": "size ✗", "ok": f"size {target} ✓"}
    for i, (pan, (why, ok)) in enumerate(zip(S["panels"], S["verdicts"])):
        t0 = 31 + 4 * i
        a = ease((k - t0) / 3)
        if a > 0:
            cx = (pan[0] + pan[2]) / 2
            col = GREEN if ok else RED
            text_center(d, (cx, pan[3] + 30), labels[why], font(22, True), with_alpha(col, a))
    a = ease((k - 31) / 4)
    if a > 0:
        text_center(d, (W / 2, 745 - 10), "Step 4: check every option — same shape, same color, size " + str(target),
                    font(22), with_alpha(GRAY, a))

    # ---- Step 5: red circle around the correct option (frames 46..59) ----
    ans = S["choices"][S["answer"]]
    pan = S["panels"][S["answer"]]
    a = (k - 46) / 11
    if a > 0:
        a = min(a, 1.0)
        r = min(pan[2] - pan[0], pan[3] - pan[1]) / 2 - 22
        cx, cy = ans["cx"], ans["cy"]
        bbox = (cx - r, cy - r, cx + r, cy + r)
        end = -90 + 360 * ease(a)
        d.arc(bbox, -90, end, fill=with_alpha(RED, 1.0), width=6)

    out = base.convert("RGBA")
    out.alpha_composite(ov)
    return out.convert("RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    S = analyse(base)
    print("sizes:", S["sizes"], "step:", S["step"], "target:", S["target"],
          "verdicts:", S["verdicts"], "answer: option", S["answer"] + 1)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(N_FRAMES):
        fr = render_frame(base, S, k)
        proc.stdin.write(np.asarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
