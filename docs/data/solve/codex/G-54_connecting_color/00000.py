from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(base)
# Each curve leaves one shape at its boundary and enters its matching partner.
curves = [
    ([(479, 289), (570, 224), (687, 223), (780, 286)], tuple(a[280, 420])),
    ([(368, 476), (408, 431), (458, 431), (505, 476)], tuple(a[476, 290])),
    ([(257, 755), (342, 667), (435, 667), (526, 778)], tuple(a[760, 190])),
]
S = 3
paths = []
for controls, color in curves:
    p = np.array(controls, dtype=float)
    t = np.linspace(0, 1, 1201)[:, None]
    points = (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t**2*p[2] + t**3*p[3]
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    paths.append((points, distances / distances[-1], color))
# Preserve all existing shape pixels, including their original edge pixels.
clear = np.all(a == 255, axis=2)
cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
       '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for frame in range(48):
    result = a.copy()
    for i, (points, arc, color) in enumerate(paths):
        progress = np.clip((frame - i*15) / 15, 0, 1)
        if progress <= 0:
            continue
        count = max(2, np.searchsorted(arc, progress, side='right'))
        pts = points[:count]
        mask = Image.new('L', (1024*S, 1024*S), 0)
        draw = ImageDraw.Draw(mask)
        xy = [tuple(p*S) for p in pts]
        draw.line(xy, fill=255, width=6*S, joint='curve')
        for x, y in [xy[0], xy[-1]]:
            draw.ellipse((x-3*S, y-3*S, x+3*S, y+3*S), fill=255)
        alpha = np.array(mask.resize(base.size, Image.Resampling.LANCZOS)).astype(float)/255
        alpha *= clear
        alpha = alpha[..., None]
        result = np.rint(result*(1-alpha) + np.array(color)*alpha).astype('uint8')
    proc.stdin.write(result.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
