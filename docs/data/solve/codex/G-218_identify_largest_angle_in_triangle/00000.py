from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(base)
mask = (np.min(a, axis=2) < 100).astype(np.uint8)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull = cv2.convexHull(max(contours, key=cv2.contourArea))
vertices = cv2.approxPolyDP(hull, 5, True).reshape(-1, 2)
assert len(vertices) == 3
angles = []
for i, p in enumerate(vertices):
    u = vertices[(i+1)%3].astype(float)-p
    v = vertices[(i+2)%3].astype(float)-p
    angles.append(math.acos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1)))
x, y = vertices[int(np.argmax(angles))]
print('Largest angle vertex:', (int(x), int(y)), 'angle:', math.degrees(max(angles)))
# The circle is drawn progressively, with brief pauses to inspect the original
# triangle and the completed answer. No other annotations alter the source.
S = 4
radius = 29
cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
       '-crf', '0', '-preset', 'slow', '-pix_fmt', 'yuv420p',
       '-movflags', '+faststart', str(OUT/'video.mp4')]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(22):
    image = base.copy()
    progress = min(1.0, max(0.0, (frame-2)/17))
    if progress > 0:
        layer = Image.new('RGBA', (1024*S,1024*S), (0,0,0,0))
        d = ImageDraw.Draw(layer)
        # Start above the vertex and trace clockwise.
        points = []
        for t in np.linspace(-math.pi/2, -math.pi/2 + 2*math.pi*progress,
                             max(2, int(240*progress))):
            points.append(((x+radius*math.cos(t))*S,(y+radius*math.sin(t))*S))
        d.line(points, fill=(230, 25, 30, 255), width=4*S, joint='curve')
        for px,py in (points[0],points[-1]):
            d.ellipse((px-2*S,py-2*S,px+2*S,py+2*S),fill=(230,25,30,255))
        layer = layer.resize(base.size, Image.Resampling.LANCZOS)
        image = Image.alpha_composite(base.convert('RGBA'), layer).convert('RGB')
    p.stdin.write(image.tobytes())
p.stdin.close()
assert p.wait() == 0
