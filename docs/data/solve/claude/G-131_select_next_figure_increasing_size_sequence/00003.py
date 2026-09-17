#!/usr/bin/env python3
"""Generate the step-by-step solution video for the 'next figure in an
increasing-size sequence' task.  Every frame starts from first_frame.png and
only adds annotation overlays; the final frame circles the correct option."""
import os, subprocess, math, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16

RED = (220, 30, 30)
BLUE = (30, 90, 200)
GREY = (110, 110, 110)
GREEN = (20, 150, 60)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------- analysis --
def analyse(img):
    """Detect coloured shapes; return sequence shapes, options, answer index."""
    a = np.asarray(img.convert("RGB")).astype(int)
    sat = (a.max(2) - a.min(2)) > 60
    lab, n = ndimage.label(sat)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 50:
            continue
        col = tuple(int(v) for v in np.median(a[ys, xs], axis=0))
        blobs.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                          cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2,
                          w=xs.max() - xs.min() + 1, h=ys.max() - ys.min() + 1,
                          area=len(xs), col=col))
    # split by the separator line (long grey horizontal line)
    grey = (~sat) & (a.max(2) < 250)
    rows = np.where(grey.sum(1) > 500)[0]
    # first cluster of long grey rows = the separator line (option boxes come later)
    first = []
    for k, r in enumerate(rows):
        if k and r - rows[k - 1] > 2:
            break
        first.append(r)
    sep_y = int(np.mean(first)) if len(first) else a.shape[0] // 2
    seq = sorted([b for b in blobs if b["cy"] < sep_y], key=lambda b: b["cx"])
    opts = sorted([b for b in blobs if b["cy"] > sep_y], key=lambda b: b["cx"])

    # dashed placeholder box: grey pixels in the top area right of last seq shape
    neutral = (np.abs(a[..., 0] - a[..., 1]) < 8) & (np.abs(a[..., 1] - a[..., 2]) < 8) \
        & (a.max(2) < 235) & (a.min(2) > 120)
    top_grey = neutral.copy(); top_grey[sep_y - 5:, :] = False
    top_grey[:, : int(seq[-1]["x1"]) + 5] = False   # right of the last sequence shape
    ys, xs = np.where(top_grey)
    box = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None

    # option frames (grey boxes) in the bottom area
    frames = []
    border = (~sat) & (a.max(2) < 215)   # box borders (200) but not their fill (238)
    for o in opts:
        col_slice = border[sep_y + 5:, int(o["cx"])]
        ys = np.where(col_slice)[0] + sep_y + 5
        row_slice = border[int(o["cy"]), :]
        xs = np.where(row_slice)[0]
        left = xs[xs < o["x0"]].max()
        right = xs[xs > o["x1"]].min()
        frames.append((left, ys.min(), right, ys.max()))

    sizes = [b["h"] for b in seq]
    steps = [sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)]
    step = sum(steps) / len(steps)
    pred = sizes[-1] + step
    ref_col = seq[-1]["col"]
    ref_ratio = seq[-1]["area"] / (seq[-1]["w"] * seq[-1]["h"])

    def score(o):
        col_d = sum(abs(o["col"][k] - ref_col[k]) for k in range(3))
        shape_d = abs(o["area"] / (o["w"] * o["h"]) - ref_ratio)
        return (col_d > 80, shape_d > 0.06, abs(o["h"] - pred))

    verdicts = []
    for o in opts:
        bad_col, bad_shape, sd = score(o)
        if bad_shape:
            verdicts.append("shape")
        elif bad_col:
            verdicts.append("color")
        elif sd > step * 0.4:
            verdicts.append("size")
        else:
            verdicts.append("ok")
    ans = min(range(len(opts)), key=lambda i: (verdicts[i] != "ok", score(opts[i])[2]))
    return dict(seq=seq, opts=opts, sizes=sizes, step=step, pred=pred, box=box,
                frames=frames, verdicts=verdicts, ans=ans, sep_y=sep_y)


# ----------------------------------------------------------------- drawing --
def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def seg(f, a, b):
    """progress 0..1 of frame f within [a, b)."""
    return ease((f - a) / max(1, (b - a)))


def hexagon(cx, cy, h):
    r = h / 2.0
    return [(cx + r * math.cos(math.radians(90 + 60 * k)),
             cy - r * math.sin(math.radians(90 + 60 * k))) for k in range(6)]


def draw_frame(base, info, f):
    img = base.copy()
    d = ImageDraw.Draw(img, "RGBA")
    seq, opts = info["seq"], info["opts"]
    f_s, f_m, f_l = font(20), font(24), font(26, True)
    cy = seq[0]["cy"]

    # --- Step 1 (frames 3-20): measure each sequence shape ---------------
    for i, b in enumerate(seq):
        t = seg(f, 3 + 4 * i, 9 + 4 * i)
        if t <= 0:
            continue
        x = b["x1"] + 12
        y0, y1 = b["y0"], b["y1"]
        yy = y0 + (y1 - y0) * t
        d.line([(x, y0), (x, yy)], fill=BLUE, width=2)
        d.line([(x - 5, y0), (x + 5, y0)], fill=BLUE, width=2)
        if t >= 1:
            d.line([(x - 5, y1), (x + 5, y1)], fill=BLUE, width=2)
            txt = f"{b['h']}"
            d.text((b["cx"], y1 + 22), txt, fill=BLUE, font=f_m, anchor="mm")
    # step arrows between shapes
    for i in range(len(seq) - 1):
        t = seg(f, 14 + 3 * i, 19 + 3 * i)
        if t <= 0:
            continue
        a, b = seq[i], seq[i + 1]
        xa, xb = a["x1"] + 30, b["x0"] - 30
        ay = cy - 80
        xe = xa + (xb - xa) * t
        d.line([(xa, ay), (xe, ay)], fill=RED, width=3)
        if t >= 1:
            d.polygon([(xb, ay), (xb - 12, ay - 7), (xb - 12, ay + 7)], fill=RED)
            d.text(((xa + xb) / 2, ay - 22), f"+{info['sizes'][i+1]-info['sizes'][i]}",
                   fill=RED, font=f_m, anchor="mm")
    if f >= 20:
        al = int(255 * seg(f, 20, 24))
        d.text((512, 70), f"Step 1: sizes grow by a constant step of about +{info['step']:.0f}",
               fill=(*RED, al), font=f_l, anchor="mm")

    # --- Step 2 (frames 24-36): predict the missing figure ----------------
    if f >= 24 and info["box"] is not None:
        t = seg(f, 24, 32)
        bx0, by0, bx1, by1 = info["box"]
        bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
        h = info["pred"] * t
        col = seq[-1]["col"]
        if h > 2:
            d.polygon(hexagon(bcx, bcy, h), outline=(*col, 255), fill=(*col, 70), width=3)
        if t >= 1:
            al = int(255 * seg(f, 32, 35))
            d.text((bcx, by1 + 22), f"≈ {info['pred']:.0f}", fill=(*BLUE, al), font=f_m, anchor="mm")
            d.text((512, 110), f"Step 2: next figure = {info['sizes'][-1]} + {info['step']:.0f} ≈ {info['pred']:.0f}, same shape & color",
                   fill=(*RED, al), font=f_l, anchor="mm")

    # --- Step 3 (frames 36-50): check each option -------------------------
    labels = {"shape": "wrong shape", "color": "wrong color", "size": "wrong size", "ok": "matches"}
    for i, (o, v) in enumerate(zip(opts, info["verdicts"])):
        t = seg(f, 36 + 3 * i, 39 + 3 * i)
        if t <= 0:
            continue
        al = int(255 * t)
        fx0, fy0, fx1, fy1 = info["frames"][i]
        fcx = (fx0 + fx1) / 2
        if v == "ok":
            d.text((fcx, fy1 + 46), f"size {o['h']} ✓ " + labels[v], fill=(*GREEN, al), font=f_s, anchor="mm")
        else:
            d.text((fcx, fy1 + 46), "✗ " + labels[v], fill=(*GREY, al), font=f_s, anchor="mm")
            # light cross-out in the corner of the option frame
            s = 14
            d.line([(fx1 - 8 - s, fy0 + 8), (fx1 - 8, fy0 + 8 + s)], fill=(*GREY, al), width=3)
            d.line([(fx1 - 8 - s, fy0 + 8 + s), (fx1 - 8, fy0 + 8)], fill=(*GREY, al), width=3)
    if f >= 46:
        al = int(255 * seg(f, 46, 49))
        d.text((512, 150), f"Step 3: only option {info['ans']+1} keeps shape, color and size step",
               fill=(*RED, al), font=f_l, anchor="mm")

    # --- Step 4 (frames 48-59): red circle sweeps around the answer ------
    if f >= 48:
        t = seg(f, 48, 58)
        fx0, fy0, fx1, fy1 = info["frames"][info["ans"]]
        cx0, cy0 = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        r = (fx1 - fx0) / 2 + 6
        bbox = [cx0 - r, cy0 - r, cx0 + r, cy0 + r]
        end = -90 + 360 * t
        if t >= 1:
            d.ellipse(bbox, outline=RED, width=6)
        else:
            d.arc(bbox, -90, end, fill=RED, width=6)
    return img.convert("RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    info = analyse(base)
    print("sizes", info["sizes"], "step", info["step"], "pred", info["pred"],
          "verdicts", info["verdicts"], "answer", info["ans"] + 1)
    frames_dir = tempfile.mkdtemp(prefix="frames_")
    for f in range(N_FRAMES):
        fr = base if f == 0 else draw_frame(base, info, f)
        fr.save(os.path.join(frames_dir, f"f{f:03d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(frames_dir, "f%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                    "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, fn))
    os.rmdir(frames_dir)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
