#!/usr/bin/env python3
"""Complete the color_cycle sequence: replace the '?' with the next element.

Sequence: orange, navy, cyan, purple, orange, ?  -> cycle of 4 -> navy.
The '?' fades out, then a navy circle (pixel-exact copy of the existing navy
circle) grows into place at the sixth position.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 25


def find_elements(a):
    """Return list of (x0, x1, y0, y1) bounding boxes of non-white column runs."""
    nonwhite = a.sum(2) < 750
    cols = nonwhite.any(0)
    segs, s = [], None
    for x in range(a.shape[1] + 1):
        on = x < a.shape[1] and cols[x]
        if on and s is None:
            s = x
        elif not on and s is not None:
            ys = np.where(nonwhite[:, s:x].any(1))[0]
            segs.append((s, x - 1, int(ys.min()), int(ys.max())))
            s = None
    return segs


def main():
    base = Image.open(SRC).convert("RGB")
    a = np.array(base).astype(int)
    segs = find_elements(a)
    circles, qmark = segs[:-1], segs[-1]

    # Determine the cycle: fill colors of the circles (sampled at centers).
    fills = [tuple(a[(y0 + y1) // 2, (x0 + x1) // 2]) for x0, x1, y0, y1 in circles]
    period = next(p for p in range(1, len(fills) + 1)
                  if all(fills[i] == fills[i - p] for i in range(p, len(fills))))
    answer_color = fills[len(fills) % period]
    src_idx = fills.index(answer_color)  # an existing circle with that color
    sx0, sx1, sy0, sy1 = circles[src_idx]
    patch = base.crop((sx0, sy0, sx1 + 1, sy1 + 1))
    pw, ph = patch.size

    # Target center: continue the horizontal spacing (coincides with the '?').
    centers = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, x1, y0, y1 in circles]
    step = (centers[-1][0] - centers[0][0]) / (len(centers) - 1)
    tcx = int(round(centers[-1][0] + step))
    tcy = int(round(np.mean([c[1] for c in centers])))
    tx0, ty0 = tcx - pw // 2, tcy - ph // 2

    # Region of the '?' to fade out.
    qx0, qx1, qy0, qy1 = qmark
    qbox = (qx0, qy0, qx1 + 1, qy1 + 1)
    qpatch = np.array(base.crop(qbox)).astype(float)
    white = np.full_like(qpatch, 255.0)

    # High-res disk mask for anti-aliased growth of the circle.
    R = pw / 2.0

    def circle_mask(scale):
        big = 8
        m = Image.new("L", (pw * big, ph * big), 0)
        d = ImageDraw.Draw(m)
        r = R * scale * big
        cx, cy = pw * big / 2, ph * big / 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        return m.resize((pw, ph), Image.LANCZOS)

    def smooth(t):
        return t * t * (3 - 2 * t)

    fade_end = 9            # '?' fully gone by this frame
    grow_start, grow_end = 8, N_FRAMES - 1

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        fr = base.copy()
        # 1) fade the question mark to white
        tq = min(1.0, i / fade_end)
        blended = qpatch * (1 - smooth(tq)) + white * smooth(tq)
        fr.paste(Image.fromarray(blended.round().astype(np.uint8)), qbox[:2])
        # 2) grow the answer circle into place
        if i >= grow_start:
            s = smooth((i - grow_start) / (grow_end - grow_start))
            if i == grow_end:
                fr.paste(patch, (tx0, ty0))
            elif s > 0:
                sw = max(1, int(round(pw * s)))
                sh = max(1, int(round(ph * s)))
                small = patch.resize((sw, sh), Image.LANCZOS)
                layer = Image.new("RGB", (pw, ph), (255, 255, 255))
                layer.paste(small, ((pw - sw) // 2, (ph - sh) // 2))
                fr.paste(layer, (tx0, ty0), circle_mask(s))
        frames.append(fr)

    # sanity: first frame identical to source
    assert np.array_equal(np.array(frames[0]), np.array(base))

    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(os.path.join(tmp, f"f{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-x264-params", "keyint=1", OUT], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"answer color {answer_color}, period {period}, placed at ({tcx},{tcy}) -> {OUT}")


if __name__ == "__main__":
    main()
