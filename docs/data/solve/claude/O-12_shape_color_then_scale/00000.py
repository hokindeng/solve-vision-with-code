#!/usr/bin/env python3
"""Generate the analogy-completion video: bottom-row oval first changes color
(color_155 -> color_246), then changes size (medium -> small), filling in the
two '?' slots. Everything else stays identical to first_frame.png."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 60

BASE = np.array(Image.open('/app/first_frame.png').convert('RGB')).astype(np.float32)

GREEN = (114, 191, 38)    # color_155
CYAN = (114, 229, 210)    # color_246
OUTLINE = (0, 0, 0)

ROW_Y = 682
SLOT2_X, SLOT3_X = 518, 854
MED_AX = (70, 35)          # medium oval semi-axes (matches bottom-left oval)
SMALL_AX = (49, 25)        # small = 0.7 * medium (same ratio as top row circles)

# Bounding regions of the two '?' glyphs (incl. dot), with a little margin.
Q2 = (498, 653, 539, 710)
Q3 = (834, 653, 875, 710)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp_color(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def oval_layer(cx, cy, ax, ay, fill):
    """Return (rgb layer, alpha mask) for an outlined oval on a blank canvas."""
    img = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([cx - ax, cy - ay, cx + ax, cy + ay], fill=fill, outline=OUTLINE, width=2)
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).ellipse([cx - ax, cy - ay, cx + ax, cy + ay], fill=255)
    return np.array(img).astype(np.float32), np.array(m).astype(np.float32) / 255.0


def whiten_region(frame, region, amount):
    x0, y0, x1, y1 = region
    sub = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = sub + (255.0 - sub) * amount


def composite(frame, layer, mask, alpha):
    a = (mask * alpha)[..., None]
    frame[:] = frame * (1 - a) + layer * a


def render(i):
    frame = BASE.copy()
    half = N_FRAMES // 2  # 30 frames per rule

    # ---- Rule 1: color change, shown in slot 2 -----------------------------
    if i >= 1:
        t = i / (half - 1)                       # 0..1 over frames 0..29
        appear = smooth(min(t / 0.25, 1.0))      # '?' fades, oval fades in
        whiten_region(frame, Q2, appear)
        col_t = smooth((t - 0.2) / 0.8)          # colour shifts after appearance starts
        col = lerp_color(GREEN, CYAN, col_t)
        layer, mask = oval_layer(SLOT2_X, ROW_Y, *MED_AX, col)
        composite(frame, layer, mask, appear)

    # ---- Rule 2: size change, shown in slot 3 ------------------------------
    if i >= half:
        t = (i - half) / (N_FRAMES - 1 - half)   # 0..1 over frames 30..59
        appear = smooth(min(t / 0.25, 1.0))
        whiten_region(frame, Q3, appear)
        s = smooth((t - 0.2) / 0.8)
        ax = int(round(MED_AX[0] + (SMALL_AX[0] - MED_AX[0]) * s))
        ay = int(round(MED_AX[1] + (SMALL_AX[1] - MED_AX[1]) * s))
        layer, mask = oval_layer(SLOT3_X, ROW_Y, ax, ay, CYAN)
        composite(frame, layer, mask, appear)

    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def main():
    import os
    os.makedirs('/app/output', exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'medium',
           '-r', str(FPS), '/app/output/video.mp4']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        f = render(i)
        if i == 0:
            assert np.array_equal(f, BASE.astype(np.uint8))
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(render(N_FRAMES - 1)).save('/app/output/last_frame.png')


if __name__ == '__main__':
    main()
