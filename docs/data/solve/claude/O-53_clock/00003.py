import math, subprocess, os
import numpy as np
from PIL import Image, ImageDraw

BASE = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
W = H = 1024
FPS = 16
N = 120

CX, CY = 512, 512
BG = (240, 248, 255)
RED = (220, 20, 60)        # minute hand
GREEN = (0, 100, 0)        # hour hand
HOUR_LEN, HOUR_W = 204, 8
MIN_LEN, MIN_W = 288, 6
DOT_R = 10

START_H, START_M = 6, 5
ELAPSED_H = 21             # 6:05 + 21h -> 3:05

base = np.array(Image.open(BASE).convert('RGB'))
red = (base[:, :, 0] > 180) & (base[:, :, 1] < 80)
green = (base[:, :, 1] > 80) & (base[:, :, 0] < 60)
face = base.copy()
face[red | green] = BG      # clock face with hands removed
face_img = Image.fromarray(face)

def hand_tip(angle_deg, length):
    a = math.radians(angle_deg)
    return (CX + length * math.sin(a), CY - length * math.cos(a))

def render(hours_elapsed):
    total_min = (START_H * 60 + START_M) + hours_elapsed * 60.0
    min_angle = (total_min % 60) * 6.0
    hour_angle = ((total_min / 60.0) % 12) * 30.0
    im = face_img.copy()
    d = ImageDraw.Draw(im)
    d.line([(CX, CY), hand_tip(hour_angle, HOUR_LEN)], fill=GREEN, width=HOUR_W)
    d.line([(CX, CY), hand_tip(min_angle, MIN_LEN)], fill=RED, width=MIN_W)
    d.ellipse([CX - DOT_R, CY - DOT_R, CX + DOT_R, CY + DOT_R], fill=(0, 0, 0))
    return im

def ease(t):  # smooth start/stop
    return 0.5 - 0.5 * math.cos(math.pi * t)

frames = []
for i in range(N):
    t = i / (N - 1)
    if i == 0:
        frames.append(Image.fromarray(base))   # exact first frame
    else:
        frames.append(render(ELAPSED_H * ease(t)))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT],
                     stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
p.stdin.close(); p.wait()
frames[-1].save('/app/output/last_frame.png')
render(0).save('/app/output/rendered_first.png')
print('wrote', OUT)
