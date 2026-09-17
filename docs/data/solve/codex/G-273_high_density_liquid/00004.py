from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Reuse the original objects exactly, and restore only their initial footprints.
xs = [130, 374, 618, 862]
surfaces = [563, 614, 596, 499]
sprites = [source.crop((x, 94, x + 67, 161)) for x in xs]
background = source.copy()
for x in xs:
    background.paste((255, 255, 255), (x, 94, x + 67, 161))

def smooth(u):
    u = max(0.0, min(1.0, u))
    return u*u*(3-2*u)

def position(t, i):
    surface = surfaces[i]
    contact = surface - 67
    # All four identical objects share gravitational acceleration in air.
    acceleration = 330.0
    impact = math.sqrt(2*(contact-94)/acceleration)
    if t <= impact:
        return 94 + 0.5*acceleration*t*t
    elapsed = t-impact
    if i == 3:
        # Drag reduces the falling speed in the lighter liquid, then the
        # object comes to rest against the interior bottom of the cup.
        duration = 4.45-impact
        u = min(1.0, elapsed/duration)
        # Gentle entry, steady descent, and a soft stop at the bottom.
        return contact + (790-contact)*(0.45*u + 0.55*smooth(u)) if u < 1 else 790
    equilibrium = surface-27
    # Entry inertia briefly immerses the object further than equilibrium;
    # buoyancy then produces a small, damped bob at the surface.
    if elapsed < 0.40:
        return contact + (surface-13-contact)*math.sin(elapsed/0.40*math.pi/2)
    q = elapsed-0.40
    if t >= 4.60:
        return equilibrium
    damping = math.exp(-2.2*q)
    fade = 1-smooth((t-4.05)/0.55)
    return equilibrium + 14*damping*math.cos(5.0*q)*fade

command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame_index in range(80):
    if frame_index == 0:
        frame = source.copy()
    else:
        frame = background.copy()
        t = frame_index/16
        for i, x in enumerate(xs):
            frame.paste(sprites[i], (x, round(position(t, i))))
    proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
