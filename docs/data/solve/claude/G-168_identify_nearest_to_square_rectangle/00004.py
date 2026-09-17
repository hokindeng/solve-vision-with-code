#!/usr/bin/env python3
"""Generate /app/output/video.mp4: find the rectangle closest to a square and circle it in red.

Step-by-step animation:
  1. frame 0            : untouched first frame
  2. frames 1..~29      : each rectangle is examined in turn; its width:height ratio is
                          written next to it (temporary annotation)
  3. frames ~30..~33    : the best ratio is emphasised
  4. frames ~34..47     : a red circle sweeps around the winning rectangle while the
                          temporary annotations fade away; last frame = scene + red circle
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES, SS = 16, 48, 4  # SS = supersampling factor for anti-aliased overlays
RED = (220, 30, 30)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def detect_rectangles(img):
    """Return list of dicts (x0,y0,x1,y1,w,h,ratio) from the black outlines."""
    arr = np.array(img.convert("RGB")).astype(int)
    black = (arr.max(axis=2) < 60)
    lab, n = ndimage.label(black)
    rects = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < 100:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w < 10 or h < 10:
            continue
        rects.append(dict(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1), w=int(w), h=int(h),
                          ratio=min(w, h) / max(w, h)))
    rects.sort(key=lambda r: (r["y0"], r["x0"]))  # reading order: top to bottom
    return rects


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def label_pos(r, W, H, tw, th):
    """Place a label just outside the rectangle, preferring above, staying inside the canvas."""
    cx = (r["x0"] + r["x1"]) / 2
    x = min(max(cx - tw / 2, 4), W - tw - 4)
    y = r["y0"] - th - 8
    if y < 4:
        y = r["y1"] + 8
    return x, y


def render_frame(base, rects, best, k):
    W, H = base.size
    n = len(rects)
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    font = ImageFont.truetype(FONT_PATH, 20 * SS)

    # timeline (frames)
    t_examine_end = 30          # per-rectangle examination 1..29
    t_pick_end = 34             # emphasise winner 30..33
    per = (t_examine_end - 1) / n

    # --- phase 1/2: annotations ------------------------------------------------
    if k >= 1:
        fade = 1.0
        if k >= t_pick_end:      # fade annotations out while the circle is drawn
            fade = 1.0 - ease((k - t_pick_end) / 7.0)
        if fade > 0:
            for i, r in enumerate(rects):
                t_start = 1 + i * per
                if k < t_start:
                    break
                a = ease((k - t_start + 1) / 2.0) * fade
                txt = f"{r['w']}:{r['h']}  ={r['ratio']:.2f}"
                bb = d.textbbox((0, 0), txt, font=font)
                tw, th = (bb[2] - bb[0]) / SS, (bb[3] - bb[1]) / SS
                x, y = label_pos(r, W, H, tw, th)
                is_best = (r is best)
                if is_best and k >= t_examine_end:
                    col = RED + (int(255 * a),)
                else:
                    col = (60, 60, 60, int(220 * a))
                d.text(((x - bb[0] / SS) * SS, (y - bb[1] / SS) * SS), txt, font=font, fill=col)
                # thin dashed frame marking the rectangle currently under examination
                cur = int(min(n - 1, (k - 1) // per)) if k < t_examine_end else -1
                if i == cur:
                    pad = 6
                    box = [(r["x0"] - pad) * SS, (r["y0"] - pad) * SS, (r["x1"] + pad) * SS, (r["y1"] + pad) * SS]
                    d.rectangle(box, outline=(90, 90, 90, int(200 * a)), width=2 * SS)

    # --- phase 3: red circle sweep ----------------------------------------------
    if k >= t_pick_end:
        prog = ease((k - t_pick_end) / (N_FRAMES - 1 - t_pick_end))
        cx, cy = (best["x0"] + best["x1"]) / 2, (best["y0"] + best["y1"]) / 2
        rad = 0.5 * np.hypot(best["w"], best["h"]) + 14
        box = [(cx - rad) * SS, (cy - rad) * SS, (cx + rad) * SS, (cy + rad) * SS]
        end = -90 + 360 * prog
        if prog >= 0.999:
            d.ellipse(box, outline=RED + (255,), width=5 * SS)
        else:
            d.arc(box, start=-90, end=end, fill=RED + (255,), width=5 * SS)

    ov = ov.resize((W, H), Image.LANCZOS)
    frame = base.convert("RGBA")
    frame.alpha_composite(ov)
    return frame.convert("RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    rects = detect_rectangles(base)
    best = max(rects, key=lambda r: r["ratio"])
    for r in rects:
        print(f"rect {r['w']}x{r['h']} at ({r['x0']},{r['y0']}) ratio={r['ratio']:.3f}"
              + ("  <-- closest to square" if r is best else ""))

    frames = [base.copy()]  # frame 0 is exactly first_frame.png
    for k in range(1, N_FRAMES):
        frames.append(render_frame(base, rects, best, k))

    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
