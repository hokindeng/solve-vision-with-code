from pathlib import Path
import math
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Extract each original object, retaining its original raster outline and colors.
mask = (source[:,:,0] > 120) & (source[:,:,0] < 180) & (source[:,:,1] < 110) & (source[:,:,2] > 150)
base = source.copy()
base[mask] = 255
objects = []
for left, right in [(100,225),(345,470),(590,715),(835,960)]:
    yy, xx = np.where(mask[:,left:right])
    xx = xx + left
    objects.append((yy, xx, source[yy,xx].copy()))

fps, frames = 16, 80
proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
initial_y = 130.5
surfaces = [472.5,472,484,635.5]
for frame in range(frames):
    t = frame / fps
    canvas = base.copy()
    for i, (yy, xx, color) in enumerate(objects):
        # Gravity accelerates each identical object equally before liquid contact.
        contact_y = surfaces[i] - 34
        acceleration = 440.0
        contact_t = math.sqrt(2*(contact_y-initial_y)/acceleration)
        if t <= contact_t:
            cy = initial_y + 0.5*acceleration*t*t
        elif i < 3:
            # Drag slows descent in the lighter liquid; settle at the cup floor.
            u = min(1., (t-contact_t)/(4.45-contact_t))
            cy = contact_y + (830-contact_y)*(1-(1-u)**1.35)
        else:
            # Buoyancy arrests the fall, followed by a small damped bob.
            dt = t-contact_t
            equilibrium = surfaces[i] + 8
            cy = equilibrium - (equilibrium-contact_y)*math.exp(-2.6*dt)*math.cos(5.2*dt)
            if t >= 4.5:
                cy = equilibrium
        shift = int(round(cy-initial_y))
        canvas[yy+shift,xx] = color
    if frame == 0:
        canvas = source
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
