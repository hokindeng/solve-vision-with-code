from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/app')
FPS, N = 16, 192
G, E, H, V = 10.1, .70, 12.5, .9
original = Image.open(ROOT / 'first_frame.png').convert('RGB')
background = original.copy()
ImageDraw.Draw(background).rectangle((399, 301, 632, 397), fill='white')
# Reuse the original ball pixels, including its one-pixel black outline.
ball = original.crop((400, 302, 471, 373))
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
green = (60, 180, 60)
# Piecewise exact ballistic trajectories. Height measures clearance of the ball
# above the ground; the supplied drawing fixes the scale at 32.16 pixels/m.
segments = []
time = 0.0
height, velocity = H, -V
while True:
    duration = (velocity + math.sqrt(velocity*velocity + 2*G*height))/G
    segments.append((time, time+duration, height, velocity))
    time += duration
    impact = velocity - G*duration
    velocity = -E*impact
    height = 0.0
    # Subpixel bounces are resolved as rest, avoiding indefinite numerical chatter.
    if velocity*velocity/(2*G)*32.16 < .18:
        break
settle_time = time

def state(t):
    for start, end, h, v in segments:
        if t < end:
            dt = t-start
            return max(0., h+v*dt-.5*G*dt*dt), v-G*dt
    return 0., 0.

(ROOT/'output').mkdir(exist_ok=True)
cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
       '-pix_fmt','rgb24','-s','1024x1024','-r',str(FPS),'-i','-',
       '-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',
       '-movflags','+faststart',str(ROOT/'output/video.mp4')]
process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(N):
    if i == 0:
        frame = original
    else:
        # A uniform slow-motion presentation spreads the complete trajectory
        # across eleven seconds, followed by a short resting interval.
        t = (i/FPS)*settle_time/11.1
        h, v = state(t)
        cy = 739 - h*32.16
        frame = background.copy()
        frame.paste(ball, (400, round(cy-35)))
        draw = ImageDraw.Draw(frame)
        if abs(v) > .015:
            length = max(9, abs(v)*6)
            x = 487
            # Center the vector beside the ball so even near-ground arrows
            # remain legible. Arrow length scales with speed.
            middle = min(cy, 770-length/2)
            sign = -1 if v > 0 else 1
            start = middle-sign*length/2
            end = middle+sign*length/2
            draw.line((x,start,x,end),fill=green,width=5)
            draw.polygon([(x,end),(x-7,end-sign*11),(x+7,end-sign*11)],fill=green)
        draw.text((508,cy-18),f'v={abs(v):.1f} m/s',font=font,fill=green)
    process.stdin.write(np.asarray(frame).tobytes())
process.stdin.close()
if process.wait():
    raise RuntimeError('ffmpeg failed')
