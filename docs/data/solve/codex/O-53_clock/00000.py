from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(source)
bg = (245, 245, 220)
hour_color = (139, 69, 19)
minute_color = (147, 112, 219)
hand_mask = np.all(a == hour_color, axis=2) | np.all(a == minute_color, axis=2)
clean = a.copy()
clean[hand_mask] = bg
yy, xx = np.indices(a.shape[:2])
hub = (np.all(a == (0, 0, 0), axis=2) & ((xx - 512)**2 + (yy - 512)**2 < 20**2))

def endpoint(angle, radius):
    theta = math.radians(angle)
    return (int(512 + radius * math.sin(theta)), int(512 - radius * math.cos(theta)))

cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
       '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
       '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p',
       '-movflags', '+faststart', str(OUT / 'video.mp4')]
process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(120):
    if frame == 0:
        result = a
    else:
        t = frame / 119
        # Continuous forward movement with gentle acceleration and deceleration.
        progress = t * t * (3 - 2 * t)
        minutes = 7 * 60 + 44 + 17 * 60 * progress
        im = Image.fromarray(clean)
        draw = ImageDraw.Draw(im)
        draw.line([(512, 512), endpoint(minutes * 6, 286.72)], fill=minute_color, width=3)
        draw.line([(512, 512), endpoint(minutes * 0.5, 204.8)], fill=hour_color, width=6)
        result = np.array(im)
        result[hub] = a[hub]
    process.stdin.write(result.tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError('ffmpeg failed')
