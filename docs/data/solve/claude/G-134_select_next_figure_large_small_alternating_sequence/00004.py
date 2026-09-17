#!/usr/bin/env python3
"""Generate the step-by-step solution video for the large/small alternation puzzle.

Frame 0 is exactly first_frame.png.  All later frames only add annotation
overlays (labels, highlight rings, check marks, and the final red circle);
every other pixel stays identical to the first frame.
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
N_FRAMES, FPS, SS = 60, 16, 4  # SS = supersampling factor for overlays

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RED = (220, 30, 30)
GREEN = (30, 160, 60)
DARK = (60, 60, 60)


# ----------------------------------------------------------------- analysis
def analyse(img):
    """Locate coloured shapes; split into sequence (top) and choices (bottom)."""
    a = np.array(img).astype(int)
    nonwhite = np.abs(a - 255).sum(2) > 30
    gray = (np.abs(a[..., 0] - a[..., 1]) < 12) & (np.abs(a[..., 1] - a[..., 2]) < 12)
    lab, n = ndimage.label(nonwhite & ~gray)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < 50:
            continue
        col = tuple(int(v) for v in np.median(a[ys, xs], axis=0))
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        fill = len(ys) / (w * h)  # circle ~0.785, pentagon ~0.69, square 1.0
        blobs.append(dict(cx=xs.mean(), cy=ys.mean(), w=w, h=h, size=max(w, h),
                          area=len(ys), color=col, fill=fill,
                          box=(xs.min(), ys.min(), xs.max(), ys.max())))
    # separator line: long gray horizontal run
    rows = np.where((nonwhite & gray).sum(1) > 500)[0]
    sep_y = int(rows.min()) if len(rows) else 700  # first long line = separator
    seq = sorted([b for b in blobs if b["cy"] < sep_y], key=lambda b: b["cx"])
    cho = sorted([b for b in blobs if b["cy"] > sep_y], key=lambda b: b["cx"])
    # question box: dashed gray square in top area, right of last seq item
    top_gray = (nonwhite & gray)
    top_gray[sep_y - 5:, :] = False
    if seq:  # ignore anything left of the last sequence item
        top_gray[:, :int(seq[-1]["box"][2]) + 5] = False
    ys, xs = np.where(top_gray)
    qbox = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    return seq, cho, qbox, sep_y


def classify(seq, cho):
    sizes = sorted(b["size"] for b in seq)
    thresh = (sizes[0] + sizes[-1]) / 2
    labels = ["LARGE" if b["size"] > thresh else "SMALL" for b in seq]
    nxt = "SMALL" if labels[-1] == "LARGE" else "LARGE"
    ref = seq[0]

    def shape_kind(b):
        return "circle" if abs(b["fill"] - 0.785) < 0.04 else "other"

    verdicts = []
    for b in cho:
        size_lab = "LARGE" if b["size"] > thresh else "SMALL"
        col_ok = max(abs(c1 - c2) for c1, c2 in zip(b["color"], ref["color"])) < 40
        shp_ok = shape_kind(b) == shape_kind(ref)
        if not shp_ok:
            verdicts.append((False, "shape"))
        elif not col_ok:
            verdicts.append((False, "color"))
        elif size_lab != nxt:
            verdicts.append((False, "size"))
        else:
            verdicts.append((True, "ok"))
    correct = [i for i, v in enumerate(verdicts) if v[0]]
    return labels, nxt, verdicts, correct[0] if correct else 0


# ----------------------------------------------------------------- drawing
def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def font(sz):
    return ImageFont.truetype(FONT, int(sz * SS))


class Overlay:
    def __init__(self, size):
        self.im = Image.new("RGBA", (size[0] * SS, size[1] * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def s(self, v):
        return v * SS

    def text(self, xy, txt, sz, color, alpha=1.0, anchor="mm"):
        if alpha <= 0:
            return
        self.d.text((self.s(xy[0]), self.s(xy[1])), txt, font=font(sz),
                    fill=color + (int(255 * alpha),), anchor=anchor)

    def ring(self, c, r, color, width, alpha=1.0, frac=1.0, start=-90):
        if alpha <= 0 or frac <= 0:
            return
        bb = [self.s(c[0] - r), self.s(c[1] - r), self.s(c[0] + r), self.s(c[1] + r)]
        col = color + (int(255 * alpha),)
        if frac >= 1:
            self.d.ellipse(bb, outline=col, width=self.s(width))
        else:
            self.d.arc(bb, start, start + 360 * frac, fill=col, width=self.s(width))
            # round caps
            for ang in (start, start + 360 * frac):
                x = c[0] + r * math.cos(math.radians(ang))
                y = c[1] + r * math.sin(math.radians(ang))
                self.d.ellipse([self.s(x - width / 2), self.s(y - width / 2),
                                self.s(x + width / 2), self.s(y + width / 2)], fill=col)

    def rrect(self, box, color, width, alpha=1.0, radius=10):
        if alpha <= 0:
            return
        self.d.rounded_rectangle([self.s(v) for v in box], radius=self.s(radius),
                                 outline=color + (int(255 * alpha),), width=self.s(width))

    def line(self, p, q, color, width, alpha=1.0):
        if alpha <= 0:
            return
        self.d.line([self.s(p[0]), self.s(p[1]), self.s(q[0]), self.s(q[1])],
                    fill=color + (int(255 * alpha),), width=self.s(width))

    def composite(self, base):
        small = self.im.resize(base.size, Image.LANCZOS)
        return Image.alpha_composite(base.convert("RGBA"), small).convert("RGB")


def render(base, k, seq, cho, qbox, labels, nxt, verdicts, correct):
    """Frame k of the animation."""
    if k == 0:
        return base.copy()
    ov = Overlay(base.size)
    t = k  # frame index

    # --- Step 1 (frames 2..22): inspect each sequence item, label its size
    for i, b in enumerate(seq):
        t0 = 2 + i * 7
        a = ease((t - t0) / 5)
        if a <= 0:
            continue
        r = b["size"] / 2 + 12
        ov.ring((b["cx"], b["cy"]), r, DARK, 2, alpha=a)
        ov.text((b["cx"], b["cy"] + r + 18), labels[i], 15, DARK, alpha=a)

    # --- Step 2 (frames 23..32): deduce the next size, write it at the ? box
    if qbox is not None:
        qcx, qcy = (qbox[0] + qbox[2]) / 2, (qbox[1] + qbox[3]) / 2
        a = ease((t - 23) / 6)
        if a > 0:
            ov.text((qcx, qbox[1] - 22), "Next: " + nxt, 17, RED, alpha=a)
            # arrow from last sequence item toward the box
            last = seq[-1]
            x0 = last["cx"] + last["size"] / 2 + 22
            x1 = qbox[0] - 8
            xe = x0 + (x1 - x0) * a
            ov.line((x0, qcy), (xe, qcy), DARK, 2)
            if a >= 1:
                ov.line((xe, qcy), (xe - 9, qcy - 7), DARK, 2)
                ov.line((xe, qcy), (xe - 9, qcy + 7), DARK, 2)

    # --- Step 3 (frames 30..46): check each option, mark tick / cross
    for i, b in enumerate(cho):
        t0 = 30 + i * 4
        a = ease((t - t0) / 3)
        if a <= 0:
            continue
        ok, why = verdicts[i]
        x, y = b["cx"], b["box"][1] - 32
        if ok:
            ov.line((x - 10, y), (x - 3, y + 8), GREEN, 4, alpha=a)
            ov.line((x - 3, y + 8), (x + 12, y - 9), GREEN, 4, alpha=a)
        else:
            ov.text((x, y), "✗ " + why, 16, DARK, alpha=a)

    # --- Step 4 (frames 46..58): red circle sweeps around the correct option
    b = cho[correct]
    frac = ease((t - 46) / 12)
    if frac > 0:
        r = b["size"] / 2 + 24
        ov.ring((b["cx"], b["cy"]), r, RED, 5, frac=frac)

    return ov.composite(base)


def main():
    base = Image.open(FIRST).convert("RGB")
    seq, cho, qbox, _ = analyse(base)
    labels, nxt, verdicts, correct = classify(seq, cho)
    print("sequence:", labels, "-> next:", nxt)
    print("verdicts:", verdicts, "-> correct option", correct + 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for k in range(N_FRAMES):
        fr = render(base, k, seq, cho, qbox, labels, nxt, verdicts, correct)
        fr.save(os.path.join(frames_dir, f"{k:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-preset", "slow", "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
