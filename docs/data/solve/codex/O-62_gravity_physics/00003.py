from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Preserve the supplied raster, including the original ball's outline.
ball = original.crop((679, 54, 755, 130))
background = original.copy()
ImageDraw.Draw(background).rectangle((679, 54, 754, 129), fill='white')
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
g, h0, elasticity = 6.0, 20.2, 0.70
first_impact = math.sqrt(2*h0/g)
impact_speed = math.sqrt(2*g*h0)
# Resolve collisions analytically so even brief late bounces do not tunnel.
segments = []
t = first_impact
speed = elasticity * impact_speed
while speed * speed / (2*g) > 0.012:
    duration = 2*speed/g
    segments.append((t, t+duration, speed))
    t += duration
    speed *= elasticity
stop_time = t

def state(physical_time):
    if physical_time < first_impact:
        return h0 - 0.5*g*physical_time**2, -g*physical_time
    for start, end, v in segments:
        if physical_time < end:
            dt = physical_time-start
            return max(0.0, v*dt-0.5*g*dt*dt), v-g*dt
    return 0.0, 0.0

command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
           '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
           '-an','-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p',
           '-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for i in range(192):
    if i == 0:
        frame = original.copy()
    else:
        # Fit the complete settling sequence into 11.4 s, then hold at rest.
        height, velocity = state((i/16) * stop_time/11.4)
        cy = round(737 - (646/h0)*height)
        frame = background.copy()
        frame.paste(ball, (679, cy-37))
        d = ImageDraw.Draw(frame)
        color = (16, 131, 146)
        x = 785
        length = abs(velocity)*8.0
        if length >= 3:
            direction = -1 if velocity > 0 else 1
            end_y = cy + direction*length
            head = min(12.0, length*0.55)
            d.line((x,cy,x,end_y-direction*head/2),fill=color,width=5)
            d.polygon([(x,end_y),(x-8,end_y-direction*head),
                       (x+8,end_y-direction*head)],fill=color)
        label_y = max(165, min(930, cy-12))
        d.text((805,label_y), f'{abs(velocity):.1f} m/s',font=font,fill=color)
    proc.stdin.write(np.asarray(frame).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
