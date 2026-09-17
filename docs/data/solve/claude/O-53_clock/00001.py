#!/usr/bin/env python3
"""Animate the clock in first_frame.png advancing 11 hours (10:04 -> 9:04)."""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
OUT = os.path.join(OUT_DIR, 'video.mp4')
FPS, N = 16, 120

BG = (240, 248, 255)
RED, BROWN = (220, 20, 60), (139, 69, 19)
CX = CY = 512
MIN_LEN, MIN_W = 286, 5
HR_LEN, HR_W = 204, 7
DOT_R = 10

START_H, START_M = 10, 4
ADVANCE_H = 11


def base_image():
    """Original frame with both hands erased (they sit on plain background)."""
    a = np.array(Image.open(SRC).convert('RGB'))
    for col in (RED, BROWN):
        a[(a == col).all(2)] = BG
    return Image.fromarray(a)


def hand_end(ang_deg, length):
    r = math.radians(ang_deg)
    return (CX + length * math.sin(r), CY - length * math.cos(r))


def draw_clock(base, minutes_total, dot_color):
    im = base.copy()
    d = ImageDraw.Draw(im)
    hr_ang = (minutes_total / 60.0) * 30.0
    min_ang = (minutes_total % 60) * 6.0
    d.line([(CX, CY), hand_end(hr_ang, HR_LEN)], fill=BROWN, width=HR_W)
    d.line([(CX, CY), hand_end(min_ang, MIN_LEN)], fill=RED, width=MIN_W)
    d.ellipse([CX - DOT_R, CY - DOT_R, CX + DOT_R, CY + DOT_R], fill=dot_color)
    return im


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = Image.open(SRC).convert('RGB')
    dot_color = tuple(int(v) for v in np.array(first)[CY, CX])
    base = base_image()
    start = START_H * 60 + START_M
    total = ADVANCE_H * 60

    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', '1024x1024', '-r', str(FPS), '-i', '-',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'medium', OUT],
        stdin=subprocess.PIPE)
    for i in range(N):
        if i == 0:
            frame = first
        else:
            t = i / (N - 1)
            frame = draw_clock(base, start + total * ease(t), dot_color)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    frame.save(os.path.join(OUT_DIR, 'last_frame.png'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
