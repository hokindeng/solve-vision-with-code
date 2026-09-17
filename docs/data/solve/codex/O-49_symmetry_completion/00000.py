from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
green = np.array([5, 150, 105], dtype=np.uint8)
# Mirror the three left columns. Existing colored cells are retained.
missing = [(1, 5), (2, 4), (3, 3), (3, 5), (5, 3), (5, 5)]
yy, xx = np.indices(base.shape[:2])
white = np.all(base == 255, axis=2)
masks = []
for row, col in missing:
    masks.append(white & (xx >= 204 + col * 102.4)
                 & (xx < 204 + (col + 1) * 102.4)
                 & (yy >= 204 + row * 102.4)
                 & (yy < 204 + (row + 1) * 102.4))
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
       '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
       '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
       '-crf', '10', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
       str(OUT / 'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(35):
    frame = base.copy()
    for j, mask in enumerate(masks):
        t = np.clip((i - j * 5.5) / 5.5, 0, 1)
        t = t * t * (3 - 2 * t)
        frame[mask] = np.rint(255 * (1 - t) + green.astype(float) * t).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
Image.fromarray(frame).save(OUT / 'last_frame.png')
