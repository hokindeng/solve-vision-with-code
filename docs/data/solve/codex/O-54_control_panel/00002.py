from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Observed mapping: middle = purple, left = green; right = orange.
purple = (75, 0, 130)
green = (0, 255, 0)
orange = (255, 165, 0)
units = [
    (204, 286, 204, purple, 1, 7),
    (434, 593, 511, green, 8, 15),
    (818, 900, 818, purple, 16, 22),
]
dot = source[674:679, 120:125].copy()
light_masks = []
for i, (_, _, _, color, _, _) in enumerate(units):
    mask = np.all(source == color, axis=2)
    xx = np.arange(1024)[None, :]
    mask &= (xx >= i * 307 + 100) & (xx < i * 307 + 310)
    light_masks.append(mask)

process = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-pixel_format', 'rgb24', '-video_size', '1024x1024', '-framerate', '16',
    '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(24):
    canvas = source.copy()
    for i, (start, end, middle, original_color, begin, finish) in enumerate(units):
        if frame < begin:
            continue
        t = min(1.0, (frame - begin + 1) / (finish - begin + 1))
        eased = t*t*(3-2*t)
        x = round(start + (end-start)*eased)
        # Reveal the track and the detent previously covered by the handle.
        canvas[648:705, start-28:start+29] = 0
        hidden_dot = start if i != 1 else 429
        canvas[674:679, hidden_dot-2:hidden_dot+3] = dot
        canvas[648:705, x-28:x+29] = 128
        if i == 1 and x < (434+middle)/2:
            color = green
        elif x < (middle+end)/2:
            color = purple
        else:
            color = orange
        canvas[light_masks[i]] = color
    process.stdin.write(canvas.tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError('Video encoding failed')
