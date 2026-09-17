#!/usr/bin/env python3
"""Animate the clock in first_frame.png advancing 16 hours (7:36 -> 11:36)."""
import math, subprocess
import numpy as np
from PIL import Image, ImageDraw

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N = 16, 120
CX, CY = 512, 512
BG = (240, 248, 255)
HOUR_COL, HOUR_LEN, HOUR_W = (75, 0, 130), 204, 8
MIN_COL, MIN_LEN, MIN_W = (65, 105, 225), 286, 5

src = np.array(Image.open(SRC).convert('RGB'))
a = src.astype(int)
hand_mask = (np.abs(a - HOUR_COL).sum(2) < 10) | (np.abs(a - MIN_COL).sum(2) < 10)
dot_mask = a.sum(2) < 100
base = src.copy()
base[hand_mask] = BG                      # clean face with hands removed

def tip(angle_deg, length):
    r = math.radians(angle_deg)
    return (CX + length * math.sin(r), CY - length * math.cos(r))

def render(hour_ang, min_ang):
    im = Image.fromarray(base.copy())
    d = ImageDraw.Draw(im)
    d.line([(CX, CY), tip(hour_ang, HOUR_LEN)], fill=HOUR_COL, width=HOUR_W)
    d.line([(CX, CY), tip(min_ang, MIN_LEN)], fill=MIN_COL, width=MIN_W)
    fr = np.array(im)
    fr[dot_mask] = src[dot_mask]          # original center dot on top
    return fr

def ease(t):  # smoothstep
    return t * t * (3 - 2 * t)

start_h = 7 + 36 / 60.0                   # 7:36
elapsed_h = 16.0
HOLD = 8                                  # frames held at start and end
frames = []
for i in range(N):
    t = min(max((i - HOLD) / float(N - 1 - 2 * HOLD), 0.0), 1.0)
    h = start_h + elapsed_h * ease(t)
    hour_ang = (h % 12) * 30
    min_ang = (h % 1) * 360
    frames.append(src if i == 0 else render(hour_ang, min_ang))

proc = subprocess.Popen(
    ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', '1024x1024', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
     '-crf', '10', '-preset', 'slow', OUT], stdin=subprocess.PIPE)
for f in frames:
    proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
proc.stdin.close(); proc.wait()
print('wrote', OUT)
