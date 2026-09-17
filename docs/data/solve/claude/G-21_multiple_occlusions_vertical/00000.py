"""Slide the gray rectangular mask straight down until it exits the frame."""
import subprocess, numpy as np
from PIL import Image

W = H = 1024
FPS, N = 16, 58
first = np.array(Image.open('/app/first_frame.png').convert('RGB'))

# Locate the mask (solid uniform rectangle) and its color.
MASK_COL = np.array([209, 209, 209], np.uint8)
m = (first == MASK_COL).all(2)
rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

# Background = first frame with the mask region restored to the background color (white).
bg = first.copy()
bg[y0:y1, x0:x1] = first[0, 0]

travel = H - y0  # distance until the top edge of the mask crosses the bottom of the frame

def frame(i):
    off = int(round(travel * i / (N - 1)))
    f = bg.copy()
    ty0, ty1 = y0 + off, min(y1 + off, H)
    if ty0 < H:
        f[ty0:ty1, x0:x1] = MASK_COL
    return f

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '12', '-preset', 'slow', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for i in range(N):
    p.stdin.write(frame(i).tobytes())
p.stdin.close(); p.wait()
