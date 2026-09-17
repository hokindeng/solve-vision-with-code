from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
background = original.copy()
pink = np.array([255, 192, 203], dtype=np.uint8)
orange = np.array([255, 140, 0], dtype=np.uint8)
background[np.all(background == pink, axis=2) | np.all(background == orange, axis=2)] = 255

# The visible wall's interior is [63, 962); account for ball radius.
radius = 30
lo, hi = 63 + radius, 962 - radius
position = np.array([607.0, 272.0])
velocity = np.array([-1.0, -1.0])
points = [position.copy()]
for _ in range(3):
    times = np.array([(lo-position[k])/velocity[k] if velocity[k] < 0
                      else (hi-position[k])/velocity[k] for k in range(2)])
    axis = int(np.argmin(times))
    position = position + velocity * times[axis]
    points.append(position.copy())
    velocity[axis] *= -1

# Allocate almost equal speed to each segment and explicitly show all impacts.
lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
impact_frames = np.rint(np.cumsum(lengths) / lengths.sum() * 76).astype(int)
knots = np.concatenate(([0], impact_frames))
proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
    '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(80):
    if frame == 0:
        canvas = original
    else:
        canvas = background.copy()
        t = min(frame, 76)
        segment = min(int(np.searchsorted(knots[1:], t, side='left')), 2)
        u = (t-knots[segment]) / (knots[segment+1]-knots[segment])
        center = points[segment]*(1-u) + points[segment+1]*u
        cv2.circle(canvas, tuple(np.rint(center).astype(int)), radius,
                   (255, 192, 203), thickness=-1, lineType=cv2.LINE_8)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Video encoding failed')
print('Collision centers:', points[1:])
