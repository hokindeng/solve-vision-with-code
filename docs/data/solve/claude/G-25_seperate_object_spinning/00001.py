#!/usr/bin/env python3
"""Rotate three squares in place to match their dashed targets, then slide them right."""
import math, subprocess, os
import numpy as np, cv2
from PIL import Image, ImageDraw

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
W = H = 1024; FPS = 16; N = 48
OUTLINE = (120, 120, 120)
DASH = (90, 90, 90)

src = np.array(Image.open(SRC).convert('RGB'))

def min_rect(mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), (w, h), a = cv2.minAreaRect(pts)
    return cx, cy, (w + h) / 2, a

def square_pts(cx, cy, side, ang_deg):
    a = math.radians(ang_deg); c, s = math.cos(a), math.sin(a); r = side / 2
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in
            ((-r, -r), (r, -r), (r, r), (-r, r))]

def draw_square(img, cx, cy, side, ang, color):
    ImageDraw.Draw(img).polygon(square_pts(cx, cy, side, ang), fill=color, outline=OUTLINE, width=1)

# --- measure objects (solid fills) ---------------------------------------
objs = []
for color in [(235, 120, 120), (173, 216, 230), (180, 100, 255)]:
    m = np.all(src == color, axis=2)
    cx, cy, side, ang = min_rect(m)
    ang %= 90
    # refine side length (+outline) by matching a re-render against a crop of the source
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min() - 6, xs.max() + 7, ys.min() - 6, ys.max() + 7
    crop = src[y0:y1, x0:x1]
    best = None
    for s in np.arange(side - 1, side + 3, 0.25):
        for dc in np.arange(-0.5, 0.51, 0.25):
            for dr in np.arange(-0.5, 0.51, 0.25):
                im = Image.new('RGB', (x1 - x0, y1 - y0), (255, 255, 255))
                draw_square(im, cx + dc - x0, cy + dr - y0, s, ang, color)
                score = np.count_nonzero(np.any(np.array(im) != crop, axis=2))
                if best is None or score < best[0]: best = (score, s, cx + dc, cy + dr)
    print('fit', color, best)
    _, side, cx, cy = best
    objs.append(dict(color=color, cx=cx, cy=cy, side=side, ang=ang))

# --- measure dashed targets ------------------------------------------------
m90 = np.all(src == DASH, axis=2)
d = cv2.dilate(m90.astype(np.uint8), np.ones((13, 13), np.uint8))
n, lab = cv2.connectedComponents(d)
targets = []
for i in range(1, n):
    mm = m90 & (lab == i)
    if mm.sum() > 200:
        cx, cy, side, ang = min_rect(mm); targets.append(dict(cx=cx, cy=cy, side=side, ang=ang % 90))

# pair each object with the target of the same row (same y) / nearest size
for o in objs:
    t = min(targets, key=lambda t: abs(t['cy'] - o['cy']) * 10 + abs(t['side'] - o['side']))
    o['tx'] = t['cx']; o['ty'] = o['cy']  # horizontal move only
    # smallest rotation (square symmetry: angles mod 90)
    da = (t['ang'] - o['ang'] + 45) % 90 - 45
    o['dang'] = da
    print(o['color'], 'side %.2f ang %.2f -> target ang %.2f (rot %.2f), move %.1f -> %.1f' %
          (o['side'], o['ang'], t['ang'], da, o['cx'], t['cx']))

# --- background: source with objects erased ----------------------------------
bg = src.copy()
erase = np.zeros((H, W), bool)
for o in objs:
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon(square_pts(o['cx'], o['cy'], o['side'] + 4, o['ang']), fill=255)
    erase |= np.array(im) > 0
bg[erase] = 255

def ease(t):  # smoothstep
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)

ROT_END = 0.42  # fraction of the timeline spent rotating; the rest translating

frames = []
for f in range(N):
    t = f / (N - 1)
    if f == 0:
        frames.append(src.copy()); continue
    im = Image.fromarray(bg.copy())
    r = ease(t / ROT_END)
    s = ease((t - ROT_END) / (1 - ROT_END))
    for o in objs:
        ang = o['ang'] + o['dang'] * r
        cx = o['cx'] + (o['tx'] - o['cx']) * s
        draw_square(im, cx, o['cy'], o['side'], ang, o['color'])
    fr = np.array(im)
    fr[m90] = DASH  # dashed outlines stay on top, pixel-identical in every frame
    frames.append(fr)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '16', OUT], stdin=subprocess.PIPE)
for fr in frames: p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('wrote', OUT)
