from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Vertices of the original triangle, in image coordinates.
vertices = np.array([[309.,149.], [224.,578.], [626.,639.]])
angles = []
for i, vertex in enumerate(vertices):
    u = vertices[(i+1)%3] - vertex
    v = vertices[(i+2)%3] - vertex
    angles.append(math.degrees(math.acos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))))
cx, cy = vertices[int(np.argmax(angles))]
print('Interior angles:', angles, 'Selected vertex:', (cx, cy))
# Leave the image alone while identifying the vertex, then progressively
# trace its red circle. Composite only the stroke, preserving all other pixels.
scale = 4
radius = 30
command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'slow',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame in range(22):
    result = base.copy()
    if frame >= 3:
        progress = min(1., (frame-2)/17.)
        overlay = Image.new('RGBA', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(overlay)
        points = []
        for angle in np.linspace(-math.pi/2, -math.pi/2+2*math.pi*progress, max(2,int(720*progress))):
            points.append(((cx+radius*math.cos(angle))*scale, (cy+radius*math.sin(angle))*scale))
        draw.line(points, fill=(230,25,35,255), width=4*scale, joint='curve')
        for x,y in [points[0],points[-1]]:
            draw.ellipse((x-2*scale,y-2*scale,x+2*scale,y+2*scale), fill=(230,25,35,255))
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        result.paste(overlay, (0,0), overlay)
    proc.stdin.write(result.tobytes())
proc.stdin.close()
stderr = proc.stderr.read()
if proc.wait():
    raise RuntimeError(stderr.decode())
