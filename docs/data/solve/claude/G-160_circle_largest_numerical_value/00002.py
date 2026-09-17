"""Generate a video that circles the largest number (90) in first_frame.png."""
import math, os, subprocess, tempfile
import numpy as np
import cv2
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 80
SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'

# Bounding box of "90" (largest of 15, 21, 90, 28, 57, 33), found via connected components
X0, Y0, X1, Y1 = 208, 366, 363, 457
CX, CY = (X0 + X1) / 2.0, (Y0 + Y1) / 2.0
RX, RY = 108.0, 72.0          # ellipse radii, clears neighbouring numbers
COLOR = (220, 30, 30)         # red (RGB)
THICK = 6

HOLD_START = 14   # frames of "comparing" before the pen starts
DRAW_END = 68     # frame at which the circle is complete
START_ANGLE = -100.0  # degrees, start near the top, sweep clockwise-ish


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def point(angle_deg):
    a = math.radians(angle_deg)
    # slight hand-drawn wobble
    w = 1.0 + 0.012 * math.sin(3 * a + 0.7) + 0.008 * math.cos(5 * a)
    return (CX + RX * w * math.cos(a), CY + RY * w * math.sin(a))


def draw_arc(img, frac):
    if frac <= 0:
        return img
    sweep = 365.0 * frac  # slight overlap at the end to close the loop
    n = max(2, int(sweep * 2))
    pts = np.array([point(START_ANGLE + sweep * i / (n - 1)) for i in range(n)])
    pts = np.round(pts * 16).astype(np.int32)
    cv2.polylines(img, [pts], False, COLOR, THICK, lineType=cv2.LINE_AA, shift=4)
    # round pen tip at the leading end
    tip = tuple(int(round(v * 16)) for v in point(START_ANGLE + sweep))
    cv2.circle(img, tip, THICK * 8, COLOR, -1, lineType=cv2.LINE_AA, shift=4)
    return img


def main():
    base = np.array(Image.open(SRC).convert('RGB'))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmpdir = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        frame = base.copy()
        if i >= HOLD_START:
            t = min(1.0, (i - HOLD_START) / float(DRAW_END - HOLD_START))
            draw_arc(frame, ease(t))
        Image.fromarray(frame).save(os.path.join(tmpdir, f'{i:04d}.png'))
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
        '-i', os.path.join(tmpdir, '%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-r', str(FPS), OUT
    ], check=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
