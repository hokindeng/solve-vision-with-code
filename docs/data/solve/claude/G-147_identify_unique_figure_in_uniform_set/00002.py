#!/usr/bin/env python3
"""Identify the unique shape among a uniform set and circle it in red, step by step."""
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 60
SS = 4  # supersampling factor for anti-aliased drawing


def detect_shapes(img):
    bg = img[0, 0].astype(int)
    mask = (np.abs(img.astype(int) - bg).sum(2) > 30).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    shapes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 50:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        fill = a / float(w * h)
        nv = len(approx)
        if fill > 0.9:
            kind = "rect"
        elif nv == 3 or fill < 0.6:
            kind = "tri"
        elif fill > 0.74:
            kind = "circle"
        else:
            kind = f"poly{nv}"
        shapes.append(dict(kind=kind, x=x, y=y, w=w, h=h, area=int(a),
                           cx=(x + w / 2.0), cy=(y + h / 2.0)))
    return shapes


def find_unique(shapes):
    # Signature = type + rounded size; the odd one has a signature seen once.
    def sig(s):
        return (s["kind"], round(s["w"] / 8.0), round(s["h"] / 8.0))
    counts = {}
    for s in shapes:
        counts[sig(s)] = counts.get(sig(s), 0) + 1
    uniq = [s for s in shapes if counts[sig(s)] == 1]
    if len(uniq) == 1:
        return uniq[0]
    # Fallback: most distant from the mean in (kind, area) space
    kinds = {}
    for s in shapes:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    odd_kind = [s for s in shapes if kinds[s["kind"]] == 1]
    if len(odd_kind) == 1:
        return odd_kind[0]
    med = np.median([s["area"] for s in shapes])
    return max(shapes, key=lambda s: abs(s["area"] - med))


def ease(t):
    return t * t * (3 - 2 * t)


def draw_arc_layer(size, cx, cy, r, deg, width, color):
    """RGBA layer with an anti-aliased partial circle (0..deg), starting at top."""
    W, H = size
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    if deg <= 0:
        return layer.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(layer)
    box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
    start = -90
    end = start + min(deg, 360)
    if deg >= 360:
        d.ellipse(box, outline=color + (255,), width=width * SS)
    else:
        d.arc(box, start, end, fill=color + (255,), width=width * SS)
        # round caps so the growing stroke looks like a pen stroke
        for ang in (start, end):
            a = np.deg2rad(ang)
            px = (cx + r * np.cos(a)) * SS
            py = (cy + r * np.sin(a)) * SS
            rr = width * SS / 2.0
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=color + (255,))
    return layer.resize((W, H), Image.LANCZOS)


def main():
    base = Image.open(SRC).convert("RGB")
    W, H = base.size
    img = np.array(base)
    shapes = detect_shapes(img)
    target = find_unique(shapes)
    print("shapes:", [(s["kind"], s["w"], s["h"]) for s in shapes])
    print("unique:", target["kind"], "at", (target["cx"], target["cy"]))

    cx, cy = target["cx"], target["cy"]
    half_diag = 0.5 * np.hypot(target["w"], target["h"])
    r = half_diag + 16
    # keep the circle within the canvas
    width = 6
    r = min(r, cx - width, cy - width, W - cx - width, H - cy - width)
    red = (220, 30, 30)

    # Timeline: hold (identify) -> draw circle -> hold finished
    hold_start = 14
    draw_frames = 34
    frames = []
    for i in range(N_FRAMES):
        if i < hold_start:
            deg = 0
        elif i < hold_start + draw_frames:
            t = (i - hold_start + 1) / float(draw_frames)
            deg = 360 * ease(t)
        else:
            deg = 360
        if deg <= 0:
            frame = base.copy()
        else:
            layer = draw_arc_layer((W, H), cx, cy, r, deg, width, red)
            frame = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
        frames.append(np.array(frame))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
