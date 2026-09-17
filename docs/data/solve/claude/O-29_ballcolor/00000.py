#!/usr/bin/env python3
"""Animate red cluster A absorbing clusters B (2), C (4), D (5) in order of
increasing size (always smaller than A), until all 14 balls are red."""
import os, subprocess, math, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, 'first_frame.png')
OUT_DIR = os.path.join(HERE, 'output')
OUT = os.path.join(OUT_DIR, 'video.mp4')
FPS, N_FRAMES = 16, 84
W = H = 1024
BG = (240, 240, 240)
SHADOW = (50, 50, 50)
RED = (255, 50, 50)
R = 26            # ball radius
STEP = 65         # grid spacing
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 48)

# cluster definitions measured from first_frame.png
CLUSTERS = {
    'A': dict(color=RED,             origin=(93, 256),  n=3, cols=2, text_xy=(202, 279)),
    'C': dict(color=(152, 251, 152), origin=(359, 509), n=4, cols=2, text_xy=(469, 531)),
    'D': dict(color=(50, 220, 220),  origin=(121, 777), n=5, cols=2, text_xy=(257, 832)),
    'B': dict(color=(0, 0, 128),     origin=(512, 831), n=2, cols=1, text_xy=(575, 853)),
}
ORDER = ['B', 'C', 'D']          # 3<->2 ok, 5>4 ok, 9>5 ok  -> 14 all red


def slot(origin, k, cols=2):
    return (origin[0] + STEP * (k % cols), origin[1] + STEP * (k // cols))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def draw_ball(d, xy, color):
    x, y = int(round(xy[0])), int(round(xy[1]))
    d.ellipse((x - R + 1, y - R + 1, x + R + 1, y + R + 1), fill=SHADOW)
    d.ellipse((x - R, y - R, x + R, y + R), fill=color)


def draw_label(img, xy, text, alpha=1.0):
    if alpha <= 0:
        return
    layer = Image.new('RGB', img.size, BG)
    d = ImageDraw.Draw(layer)
    bb = d.textbbox(xy, text, font=FONT)
    d.rectangle((bb[0] - 3, bb[1] - 3, bb[2] + 3, bb[3] + 3), fill=(255, 255, 255))
    d.text(xy, text, font=FONT, fill=(0, 0, 0))
    if alpha >= 1:
        box = (bb[0] - 3, bb[1] - 3, bb[2] + 4, bb[3] + 4)
        img.paste(layer.crop(box), box[:2])
    else:
        mask = Image.new('L', img.size, 0)
        ImageDraw.Draw(mask).rectangle((bb[0] - 3, bb[1] - 3, bb[2] + 3, bb[3] + 3), fill=int(255 * alpha))
        img.paste(layer, (0, 0), mask)


def make_base():
    """First frame with all dynamic elements (balls + cluster labels) erased."""
    img = Image.open(FIRST).convert('RGB')
    d = ImageDraw.Draw(img)
    for c in CLUSTERS.values():
        for k in range(c['n']):
            x, y = slot(c['origin'], k, c['cols'])
            d.rectangle((x - R - 2, y - R - 2, x + R + 3, y + R + 3), fill=BG)
        bb = d.textbbox(c['text_xy'], f"X: {c['n']}", font=FONT)
        d.rectangle((bb[0] - 6, bb[1] - 6, bb[2] + 12, bb[3] + 6), fill=BG)
    return img


# ---- timeline -------------------------------------------------------------
PHASE_LEN = 27                   # frames per absorption (1..81), 82/83 hold
HOLD0, TRAVEL = 3, 18            # inside a phase: hold, travel, then settle


def state_at(frame):
    """Return list of (xy, color) balls and list of (xy, text, alpha) labels."""
    balls, labels = [], []
    a_count = CLUSTERS['A']['n']
    for k in range(a_count):
        balls.append((slot(CLUSTERS['A']['origin'], k), RED))
    a_display = a_count
    for i, name in enumerate(ORDER):
        c = CLUSTERS[name]
        start = 1 + i * PHASE_LEN
        t_local = frame - start
        if t_local < HOLD0:
            prog = 0.0
        else:
            prog = (t_local - HOLD0) / TRAVEL
        prog = min(max(prog, 0.0), 1.0)
        assert a_count > c['n']
        for k in range(c['n']):
            src = slot(c['origin'], k, c['cols'])
            dst = slot(CLUSTERS['A']['origin'], a_count + k)
            e = ease(prog)
            xy = (src[0] + (dst[0] - src[0]) * e, src[1] + (dst[1] - src[1]) * e)
            col = lerp(c['color'], RED, ease(max(0.0, (prog - 0.55) / 0.45)))
            balls.append((xy, col))
        alpha = 1.0 - ease(min(1.0, prog / 0.6))
        labels.append((c['text_xy'], f"{name}: {c['n']}", alpha))
        a_count += c['n']
        if prog >= 1.0:
            a_display = a_count
    labels.append((CLUSTERS['A']['text_xy'], f"A: {a_display}", 1.0))
    return balls, labels


def render(frame, base):
    img = base.copy()
    d = ImageDraw.Draw(img)
    balls, labels = state_at(frame)
    for xy, col in balls:
        draw_ball(d, xy, col)
    for xy, text, alpha in labels:
        draw_label(img, xy, text, alpha)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = make_base()
    f0 = np.array(render(0, base)).astype(int)
    ref = np.array(Image.open(FIRST).convert('RGB')).astype(int)
    diff = np.abs(f0 - ref).sum(axis=2)
    print('frame0 mismatched pixels:', int((diff > 0).sum()))
    frames_dir = os.path.join(OUT_DIR, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(N_FRAMES):
        render(f, base).save(os.path.join(frames_dir, f'{f:04d}.png'))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                    '-i', os.path.join(frames_dir, '%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', OUT], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
