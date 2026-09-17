#!/usr/bin/env python3
"""Delete the red hollow diamond at position 4: fade it out, then slide symbols 5 and 6 left."""
import subprocess, numpy as np
from PIL import Image

W = H = 1024; FPS = 16; N = 45
FADE_END = 18           # frames 0..17 fade, 18..44 slide
PITCH = 105             # distance between cell origins
# cell interiors (exclusive of the gray 1px borders): x0 = border+1, width 94
CELL_X = [203, 308, 413, 518, 623, 728]
Y0, Y1 = 465, 560       # interior rows [465, 560)
CW = 94

base = np.array(Image.open('/app/first_frame.png').convert('RGB')).astype(np.float32)

def interior(i):
    x = CELL_X[i]
    return base[Y0:Y1, x:x + CW].copy()

patch4, patch5, patch6 = interior(3), interior(4), interior(5)

# background with cells 4,5,6 emptied (interiors are pure white)
bg = base.copy()
for i in (3, 4, 5):
    bg[Y0:Y1, CELL_X[i]:CELL_X[i] + CW] = 255

def multiply(frame, patch, x):
    """Composite a white-backed patch onto the frame by multiplication (glyph over anything)."""
    xi = int(round(x))
    region = frame[Y0:Y1, xi:xi + CW]
    frame[Y0:Y1, xi:xi + CW] = region * (patch / 255.0)

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

frames = []
for f in range(N):
    fr = bg.copy()
    if f < FADE_END:
        a = 1.0 - ease(f / (FADE_END - 1))          # 1 -> 0 visibility
        faded = 255 + (patch4 - 255) * a
        multiply(fr, faded, CELL_X[3])
        dx = 0.0
    else:
        t = (f - FADE_END) / (N - 1 - FADE_END)     # 0 -> 1
        dx = PITCH * ease(t)
    multiply(fr, patch5, CELL_X[4] - dx)
    multiply(fr, patch6, CELL_X[5] - dx)
    frames.append(np.clip(fr + 0.5, 0, 255).astype(np.uint8))

frames[0] = base.astype(np.uint8)  # exact first frame

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '16', '-preset', 'medium', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('wrote /app/output/video.mp4', len(frames), 'frames')
