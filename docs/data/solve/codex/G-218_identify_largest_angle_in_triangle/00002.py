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
# The three line junctions, in image coordinates.
vertices = np.array([[418., 252.], [837., 346.], [319., 560.]])
angles = []
for i, v in enumerate(vertices):
    a, b = vertices[(i+1)%3] - v, vertices[(i+2)%3] - v
    angles.append(math.acos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1)))
center = vertices[int(np.argmax(angles))]
# Pause briefly to examine the triangle, then draw the answer in one stroke.
# Composite only the red stroke, leaving the source pixels elsewhere untouched.
frames = []
scale = 4
for index in range(22):
    frame = base.copy()
    progress = np.clip((index - 2) / 17, 0, 1)
    if progress > 0:
        mask = Image.new('L', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(mask)
        theta = np.linspace(-math.pi/2, -math.pi/2 + 2*math.pi*progress, max(2, int(240*progress)))
        points = [((center[0]+30*math.cos(t))*scale, (center[1]+30*math.sin(t))*scale) for t in theta]
        draw.line(points, fill=255, width=4*scale, joint='curve')
        for x,y in (points[0], points[-1]):
            draw.ellipse((x-2*scale,y-2*scale,x+2*scale,y+2*scale),fill=255)
        mask = mask.resize(base.size, Image.Resampling.LANCZOS)
        frame.paste((230, 24, 30), (0,0), mask)
    frames.append(np.asarray(frame))
proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for frame in frames:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
print('Angles:', [round(math.degrees(a),2) for a in angles])
