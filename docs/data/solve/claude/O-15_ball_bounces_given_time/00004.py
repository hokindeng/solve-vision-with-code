#!/usr/bin/env python3
"""Simulate a ball bouncing 6 times off the boundary walls (elastic reflection)."""
import subprocess, math
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 80
N_BOUNCES = 6
STOP_FRAME = 76          # frame index at which the 6th collision happens; ball rests afterwards

BALL_COLOR = (138, 43, 226)
ARROW_COLOR = (255, 140, 0)
R = 30
CX0, CY0 = 732.0, 435.0        # measured ball centre in first frame
DIR = np.array([150.0, -58.0])  # measured arrow direction (centre -> tip)
DIR /= np.linalg.norm(DIR)

# inner white area is x,y in [63, 961]; ball centre must stay within R of it
XMIN, XMAX = 63 + R, 961 - R
YMIN, YMAX = 63 + R, 961 - R


def simulate_path():
    """Return list of centre positions (waypoints) from start through the 6th wall hit."""
    pos = np.array([CX0, CY0])
    vel = DIR.copy()
    pts = [pos.copy()]
    for _ in range(N_BOUNCES):
        tx = ((XMAX if vel[0] > 0 else XMIN) - pos[0]) / vel[0] if vel[0] != 0 else math.inf
        ty = ((YMAX if vel[1] > 0 else YMIN) - pos[1]) / vel[1] if vel[1] != 0 else math.inf
        t = min(tx, ty)
        pos = pos + vel * t
        if tx <= ty:
            pos[0] = XMAX if vel[0] > 0 else XMIN
            vel[0] = -vel[0]
        if ty <= tx:
            pos[1] = YMAX if vel[1] > 0 else YMIN
            vel[1] = -vel[1]
        pts.append(pos.copy())
    return pts


def position_at(pts, s):
    """Position after travelling arc length s along the polyline pts."""
    for a, b in zip(pts[:-1], pts[1:]):
        seg = np.linalg.norm(b - a)
        if s <= seg:
            return a + (b - a) * (s / seg)
        s -= seg
    return pts[-1].copy()


def main():
    first = Image.open('/app/first_frame.png').convert('RGB')
    arr = np.array(first)
    # background: first frame with ball and arrow removed (they sit on pure white)
    bg = arr.copy()
    for col in (BALL_COLOR, ARROW_COLOR):
        bg[np.all(arr == col, axis=2)] = (255, 255, 255)
    bg_img = Image.fromarray(bg)

    pts = simulate_path()
    total = sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:]))

    frames = [first]
    for i in range(1, N_FRAMES):
        s = total * min(i, STOP_FRAME) / STOP_FRAME
        cx, cy = position_at(pts, s)
        cx, cy = int(round(cx)), int(round(cy))
        img = bg_img.copy()
        ImageDraw.Draw(img).ellipse((cx - R, cy - R, cx + R, cy + R), fill=BALL_COLOR)
        frames.append(img)

    raw = b''.join(np.array(f, dtype=np.uint8).tobytes() for f in frames)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow',
           '-frames:v', str(N_FRAMES), '/app/output/video.mp4']
    subprocess.run(cmd, input=raw, check=True)
    print('waypoints:', [tuple(np.round(p, 1)) for p in pts])


if __name__ == '__main__':
    main()
