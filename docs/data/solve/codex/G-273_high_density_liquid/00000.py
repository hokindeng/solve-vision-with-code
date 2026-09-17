from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Extract the exact original objects so their shape, color, and outlines persist.
orange = (source[:, :, 0] > 200) & (source[:, :, 1] > 60) & (source[:, :, 1] < 180) & (source[:, :, 2] < 30)
background = source.copy()
background[orange] = 255
objects = []
for left, right in [(100, 220), (350, 470), (590, 710), (830, 960)]:
    yy, xx = np.where(orange[:, left:right])
    x0, x1 = left + xx.min(), left + xx.max() + 1
    y0, y1 = yy.min(), yy.max() + 1
    objects.append((x0, y0, source[y0:y1, x0:x1].copy(), orange[y0:y1, x0:x1].copy()))

surfaces = [522, 491, 528, 611]
command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame in range(80):
    canvas = background.copy()
    t = frame / 16
    for i, (x0, y0, sprite, mask) in enumerate(objects):
        h, w = mask.shape
        initial_center = y0 + (h - 1) / 2
        contact_center = surfaces[i] - (h - 1) / 2
        # Acceleration in air, then slower travel through the liquid.
        start = 0.25
        fall_duration = 1.35 if i < 3 else 1.50
        u = np.clip((t - start) / fall_duration, 0, 1)
        center = initial_center + (contact_center - initial_center) * u * u
        if t > start + fall_duration:
            elapsed = t - start - fall_duration
            if i < 3:
                end_center = 855 - (h - 1) / 2
                v = np.clip(elapsed / 2.85, 0, 1)
                center = contact_center + (end_center - contact_center) * (1 - (1 - v) ** 1.35)
            else:
                # The dense green liquid supports the object, partly submerged.
                final_center = surfaces[i] + 3
                v = np.clip(elapsed / 2.4, 0, 1)
                center = final_center - (final_center - contact_center) * np.exp(-6*v) * np.cos(9*v)
                if v == 1:
                    center = final_center
        top = int(round(center - (h - 1) / 2))
        patch = canvas[top:top+h, x0:x0+w]
        patch[mask] = sprite[mask]
    if frame == 0:
        canvas = source.copy()
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
