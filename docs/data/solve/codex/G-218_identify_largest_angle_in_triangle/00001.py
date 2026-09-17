from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Center-line intersections of the three original triangle sides.
vertices = np.array([[569.,269.], [743.,305.], [812.,464.]])
angles = []
for i, vertex in enumerate(vertices):
    a = vertices[(i+1)%3] - vertex
    b = vertices[(i+2)%3] - vertex
    angles.append(np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
cx, cy = vertices[np.argmax(angles)]
print('Interior angles:', np.degrees(angles))
# First observe the triangle, then trace the answer clockwise, then hold it.
# No labels or extra marks are added: only the requested red circle changes.
scale = 4
encoder = subprocess.Popen([
    'ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
    '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
    '-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
    '-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for frame in range(22):
    result = base.copy()
    if frame >= 2:
        progress = min(1., (frame-1)/18.)
        overlay = Image.new('RGBA',(1024*scale,1024*scale))
        draw = ImageDraw.Draw(overlay)
        theta = np.linspace(-np.pi/2, -np.pi/2 + progress*2*np.pi, max(2,int(360*progress)))
        points = [((cx+24*np.cos(t))*scale, (cy+24*np.sin(t))*scale) for t in theta]
        draw.line(points, fill=(230,25,30,255), width=3*scale, joint='curve')
        for x,y in [points[0],points[-1]]:
            r=1.5*scale
            draw.ellipse((x-r,y-r,x+r,y+r),fill=(230,25,30,255))
        overlay = overlay.resize(base.size,Image.Resampling.LANCZOS)
        result.paste(overlay,(0,0),overlay)
    encoder.stdin.write(np.asarray(result).tobytes())
encoder.stdin.close()
if encoder.wait() != 0:
    raise RuntimeError('Video encoding failed')
