#!/usr/bin/env python3
"""Generate the analogy-completion video: fade out '?', grow the scaled rhombus."""
import subprocess, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

W = H = 1024
FPS, N = 16, 60
FIRST = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
SS = 4  # supersampling for antialiased drawing


def measure(img):
    a = np.asarray(img).astype(int)
    orange = (a[:, :, 0] > 200) & (abs(a[:, :, 1] - 128) < 40) & (a[:, :, 2] < 60)
    dark = a.sum(2) < 400
    lab, n = ndimage.label(orange | dark)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        comps.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                          cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2,
                          n=len(xs), orange=int(orange[lab == i].sum())))
    shapes = [c for c in comps if c['orange'] > 1000]
    shapes.sort(key=lambda c: (c['cy'], c['cx']))
    A, B, C = shapes[0], shapes[1], shapes[2]
    # gray question mark
    gray = (abs(a[:, :, 0] - a[:, :, 1]) < 6) & (a[:, :, 0] < 200) & (a[:, :, 0] > 50)
    ys, xs = np.where(gray)
    qbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    qcx = (xs.min() + xs.max()) / 2
    return A, B, C, qbox, qcx


def rhombus(cx, cy, hw, hh):
    return [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    base = Image.open(FIRST).convert('RGB')
    A, B, C, qbox, qcx = measure(base)
    # scale factor from the hexagon pair (outer widths, outline included)
    k = (B['x1'] - B['x0']) / (A['x1'] - A['x0'])
    # rhombus C half extents (outer edge -> polygon vertex, minus half outline)
    hw = ((C['x1'] - C['x0']) / 2 + 1.5) * k  # +outline allowance
    hh = ((C['y1'] - C['y0']) / 2 + 1.5) * k
    # target center: horizontally at the '?' / B column, vertically aligned with C
    cx, cy = (B['cx'] + qcx) / 2, C['cy']

    # background with the '?' removed
    clean = base.copy()
    ImageDraw.Draw(clean).rectangle(qbox, fill=(255, 255, 255))
    base_np = np.asarray(base).astype(float)
    clean_np = np.asarray(clean).astype(float)

    frames = []
    for i in range(N):
        t = i / (N - 1)
        # phase 1 (0-30%): '?' fades out; phase 2 (20%-100%): rhombus grows
        f = np.clip(t / 0.3, 0, 1)
        bg = base_np * (1 - f) + clean_np * f
        frame = Image.fromarray(np.round(bg).astype(np.uint8))
        s = ease(np.clip((t - 0.2) / 0.8, 0, 1))
        if s > 0:
            big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
            d = ImageDraw.Draw(big)
            pts = [(x * SS, y * SS) for x, y in rhombus(cx, cy, hw * s, hh * s)]
            d.polygon(pts, fill=(255, 128, 0, 255), outline=(0, 0, 0, 255), width=SS)
            layer = big.resize((W, H), Image.LANCZOS)
            frame = Image.alpha_composite(frame.convert('RGBA'), layer).convert('RGB')
        frames.append(np.asarray(frame))
    frames[0] = np.asarray(base)  # exact first frame

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '0', OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()
    print(f'k={k:.4f} target center=({cx:.1f},{cy:.1f}) half extents=({hw:.1f},{hh:.1f}) -> {OUT}')


if __name__ == '__main__':
    main()
