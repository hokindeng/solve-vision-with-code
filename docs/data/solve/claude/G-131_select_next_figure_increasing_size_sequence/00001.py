#!/usr/bin/env python3
"""Generate /app/output/video.mp4: solve the "next figure in an increasing-size
sequence" puzzle shown in /app/first_frame.png, step by step.

Every frame is first_frame.png plus overlay annotations only (labels, step
arrows, verdict marks and the final red circle). Nothing else is altered.
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 60

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BLUE = (30, 90, 200)
GREEN = (20, 140, 60)
RED = (220, 30, 30)
GREY = (90, 90, 90)


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Scene analysis
# ----------------------------------------------------------------------------
def classify_shape(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    n = len(approx)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    fill = area / float(w * h)
    if n == 3:
        return "triangle"
    if n == 4:
        return "square" if fill > 0.9 else "diamond"
    if n == 5:
        return "pentagon"
    if n == 6:
        return "hexagon"
    if fill > 0.75:
        return "circle"
    return "star" if n > 6 else "polygon%d" % n


def analyse(img):
    a = np.asarray(img).astype(int)
    sat = (a.max(2) - a.min(2)) > 40          # coloured (non grey) pixels
    n, lab, stats, cent = cv2.connectedComponentsWithStats(sat.astype(np.uint8), 8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 200:
            continue
        m = lab == i
        col = tuple(int(v) for v in a[m].mean(0))
        shapes.append(dict(x=x, y=y, w=w, h=h, size=max(w, h),
                           cx=x + w / 2.0, cy=y + h / 2.0, color=col,
                           shape=classify_shape(m)))

    # separator line: long horizontal run of non-white grey pixels
    grey = (~sat) & (a.max(2) < 250)
    rows = grey.sum(1)
    sep_y = int(np.argmax(rows))

    seq = sorted([s for s in shapes if s["cy"] < sep_y], key=lambda s: s["cx"])
    choices = sorted([s for s in shapes if s["cy"] > sep_y], key=lambda s: s["cx"])

    # choice panels: connected components of the light grey (238) colour
    panel_mask = (np.abs(a - 238).sum(2) < 6)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(panel_mask.astype(np.uint8), 8)
    panels = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > 5000 and y > sep_y:
            panels.append((x, y, w, h))
    panels.sort()
    # some panels are split by the shape; merge by x overlap
    merged = []
    for p in panels:
        if merged and abs(merged[-1][0] - p[0]) < 10:
            continue
        merged.append(p)
    panels = merged
    for c in choices:
        best = min(panels, key=lambda p: abs(p[0] + p[2] / 2 - c["cx"]))
        c["panel"] = best

    # dashed placeholder box (colour ~200 grey) in the sequence area
    dash = grey.copy()
    dash[sep_y - 3:, :] = False
    ys, xs = np.where(dash)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None

    sizes = [s["size"] for s in seq]
    steps = [b - a_ for a_, b in zip(sizes, sizes[1:])]
    step = int(round(float(np.median(steps))))
    target = sizes[-1] + step
    ref_shape = seq[-1]["shape"]
    ref_col = np.array(seq[-1]["color"])

    def ok(c):
        return (c["shape"] == ref_shape
                and np.abs(np.array(c["color"]) - ref_col).sum() < 60
                and abs(c["size"] - target) <= max(4, step // 4))

    scores = [(abs(c["size"] - target) if c["shape"] == ref_shape and
               np.abs(np.array(c["color"]) - ref_col).sum() < 60 else 10 ** 6)
              for c in choices]
    answer = int(np.argmin(scores))
    for i, c in enumerate(choices):
        c["ok"] = (i == answer)
        if c["shape"] != ref_shape:
            c["why"] = "shape"
        elif np.abs(np.array(c["color"]) - ref_col).sum() >= 60:
            c["why"] = "color"
        elif abs(c["size"] - target) > max(4, step // 4):
            c["why"] = "size %d" % c["size"]
        else:
            c["why"] = "size %d" % c["size"]
    return dict(seq=seq, choices=choices, step=step, target=target, box=box,
                sep_y=sep_y, answer=answer, sizes=sizes)


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def text_center(d, xy, s, f, fill):
    bbox = d.textbbox((0, 0), s, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((xy[0] - w / 2 - bbox[0], xy[1] - h / 2 - bbox[1]), s, font=f, fill=fill)


def draw_arrow(d, p0, p1, fill, width=3, head=10):
    d.line([p0, p1], fill=fill, width=width)
    v = np.array(p1) - np.array(p0)
    L = np.linalg.norm(v)
    if L < 1:
        return
    v = v / L
    n = np.array([-v[1], v[0]])
    tip = np.array(p1)
    a = tip - v * head + n * head * 0.5
    b = tip - v * head - n * head * 0.5
    d.polygon([tuple(tip), tuple(a), tuple(b)], fill=fill)


def render_frame(base, info, k):
    """Return frame k (0..N_FRAMES-1) as a PIL image."""
    img = base.copy()
    if k == 0:
        return img
    d = ImageDraw.Draw(img)
    f_big = font(30)
    f_mid = font(24)
    f_small = font(20)
    seq, choices = info["seq"], info["choices"]
    step, target, box = info["step"], info["target"], info["box"]

    # ---- Phase 1 (frames 1-15): measure each sequence shape -------------
    for i, s in enumerate(seq):
        t0 = 1 + i * 4
        if k >= t0:
            pad = 6
            d.rectangle([s["x"] - pad, s["y"] - pad, s["x"] + s["w"] + pad,
                         s["y"] + s["h"] + pad], outline=BLUE, width=2)
            text_center(d, (s["cx"], s["y"] + s["h"] + 28), "size %d" % s["size"],
                        f_small, BLUE)

    # ---- Phase 1b (frames 13-21): step between consecutive shapes --------
    for i in range(len(seq) - 1):
        t0 = 13 + i * 4
        if k >= t0:
            a, b = seq[i], seq[i + 1]
            y = min(a["y"], b["y"]) - 30
            x0, x1 = a["x"] + a["w"] + 8, b["x"] - 8
            draw_arrow(d, (x0, y), (x1, y), GREEN, width=3, head=10)
            text_center(d, ((x0 + x1) / 2, y - 18), "+%d" % step, f_mid, GREEN)

    # ---- Phase 2 (frames 21-31): the rule and the predicted next size ----
    if k >= 21:
        top_y = min(s["y"] for s in seq) - 110
        text_center(d, (512, top_y), "Same shape & color, size step = +%d" % step,
                    f_big, GREY)
    if k >= 25 and box is not None:
        bx = (box[0] + box[2]) / 2
        by = (box[1] + box[3]) / 2
        last = seq[-1]
        y = last["y"] - 30
        draw_arrow(d, (last["x"] + last["w"] + 8, y), (box[0] - 8, y), GREEN, 3, 10)
        text_center(d, ((last["x"] + last["w"] + box[0]) / 2, y - 18), "+%d" % step,
                    f_mid, GREEN)
        text_center(d, (bx, by - 14), "%d + %d" % (seq[-1]["size"], step), f_small, GREY)
        text_center(d, (bx, by + 16), "= %d" % target, f_mid, BLUE)
    if k >= 29:
        yb = info["sep_y"] - 30
        text_center(d, (512, yb),
                    "Looking for: %s, same color, size %d" % (seq[-1]["shape"], target),
                    f_mid, GREY)

    # ---- Phase 3 (frames 31-46): check the four options -----------------
    for i, c in enumerate(choices):
        t0 = 31 + i * 4
        if k < t0:
            continue
        px, py, pw, ph = c["panel"]
        cx = px + pw / 2
        label_y = py + ph + 22
        if c["ok"]:
            text_center(d, (cx + 14, label_y), "%s  size %d" % (c["shape"], c["size"]),
                        f_small, GREEN)
            # check mark to the left of the label
            x0, y0 = px + 4, label_y
            d.line([(x0, y0), (x0 + 7, y0 + 8), (x0 + 20, y0 - 8)], fill=GREEN, width=4)
        else:
            why = c["why"]
            if why == "shape":
                msg = "wrong shape"
            elif why == "color":
                msg = "wrong color"
            else:
                msg = "wrong %s" % why
            text_center(d, (cx, label_y), msg, f_small, RED)
            x0, y0 = px + 14, py + 14
            d.line([(x0, y0), (x0 + 22, y0 + 22)], fill=RED, width=4)
            d.line([(x0 + 22, y0), (x0, y0 + 22)], fill=RED, width=4)

    # ---- Phase 4 (frames 46-59): red circle around the answer -----------
    if k >= 46:
        c = choices[info["answer"]]
        px, py, pw, ph = c["panel"]
        cx, cy = px + pw / 2, py + ph / 2
        r = max(pw, ph) / 2 + 8
        prog = ease((k - 46) / 10.0)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        if prog >= 1.0:
            d.ellipse(bbox, outline=RED, width=6)
        else:
            d.arc(bbox, start=-90, end=-90 + 360 * prog, fill=RED, width=6)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    info = analyse(base)
    print("sequence sizes:", info["sizes"], "step:", info["step"], "target:", info["target"])
    for i, c in enumerate(info["choices"]):
        print("option %d: %s %s size %d -> %s" % (i + 1, c["shape"], c["color"], c["size"],
                                                 "CORRECT" if c["ok"] else c["why"]))
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for k in range(N_FRAMES):
        fr = render_frame(base, info, k)
        fr.save(os.path.join(frames_dir, "f%03d.png" % k))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(frames_dir, "f%03d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
           "-preset", "slow", "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
