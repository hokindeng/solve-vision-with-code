from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
xs = [44 + 105*i for i in range(9)]
y0, y1 = 465, 560
# Copy the original artwork, including its exact colors and edge pixels.
sprites = [base[y0:y1, x+1:x+96].copy() for x in xs[:5]]
allowed = np.zeros(base.shape[:2], dtype=bool)
for x in xs:
    allowed[y0:y1, x+1:x+96] = True

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t*t*(3-2*t)

def render(frame):
    if frame < 6:
        return base.copy()
    canvas = base.copy()
    # Only the sequence artwork changes; grid, labels, and reference stay fixed.
    for x in xs[:7]:
        canvas[y0:y1, x+1:x+96] = 255
    def put(sprite, pos, opacity=1.0):
        x = int(round(xs[0]+1+105*pos))
        region = canvas[y0:y1, x:x+95]
        ink = np.any(sprite < 250, axis=2)
        mask = ink & allowed[y0:y1, x:x+95]
        blended = np.rint(region*(1-opacity)+sprite*opacity).astype(np.uint8)
        region[mask] = blended[mask]
    shift1 = smooth((frame-6)/13)
    put(sprites[0], 0)
    for original_pos in range(1, 5):
        pos = original_pos + shift1
        if original_pos == 4:
            pos += smooth((frame-30)/10)
        put(sprites[original_pos], pos)
    if frame >= 20:
        put(sprites[0], 1, smooth((frame-20)/7))
    if frame >= 41:
        put(sprites[0], 5, smooth((frame-41)/8))
    return canvas

command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
           '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame in range(54):
    pixels = render(frame)
    assert np.array_equal(pixels[~allowed], base[~allowed])
    proc.stdin.write(pixels.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
assert np.array_equal(render(0), base)
