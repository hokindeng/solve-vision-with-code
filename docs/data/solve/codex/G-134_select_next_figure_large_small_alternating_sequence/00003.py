from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_path, 23)
red = (220, 35, 42)

# Each successive size observation remains visible so the alternating rule
# is established before the answer is marked.
observations = [(6, 241, 'SMALL'), (16, 421, 'LARGE'),
                (26, 602, 'SMALL'), (36, 782, 'LARGE')]
command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
           '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
           '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame in range(60):
    im = base.copy()
    draw = ImageDraw.Draw(im)
    for start, x, label in observations:
        if frame >= start:
            draw.text((x, 427), label, font=font, fill=red, anchor='mt')
    if frame >= 43:
        # Draw the red answer circle progressively around choice three.
        progress = min(1.0, (frame - 42) / 13)
        points = []
        for i in range(int(240 * progress) + 1):
            angle = -math.pi/2 + 2*math.pi * i/240
            points.append((626 + 85*math.cos(angle), 855 + 87*math.sin(angle)))
        if len(points) > 1:
            draw.line(points, fill=red, width=5, joint='curve')
    proc.stdin.write(np.asarray(im, dtype=np.uint8).tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait() != 0:
    raise RuntimeError(err.decode())
print(OUT / 'video.mp4')
