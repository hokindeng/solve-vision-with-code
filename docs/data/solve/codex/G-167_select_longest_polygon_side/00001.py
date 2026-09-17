from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Vertices follow the original outline; compare the horizontal side last.
vertices = [(895, 430), (685, 758), (554, 729), (517, 624), (235, 430)]
edges = list(zip(vertices, vertices[1:] + vertices[:1]))
lengths = [math.dist(a, b) for a, b in edges]
longest = int(np.argmax(lengths))
a, b = edges[longest]
mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
print('Edge lengths in pixels:', [round(x, 1) for x in lengths])
print('Longest edge midpoint:', mid)

# Each comparison traces one edge. The temporary trace is removed before
# marking the answer, so the final image contains only the requested circle.
def frame(i):
    if i == 0:
        return base.copy()
    overlay = Image.new('RGBA', (4096, 4096))
    draw = ImageDraw.Draw(overlay)
    if 1 <= i <= 15:
        edge_index = (i - 1) // 3
        fraction = ((i - 1) % 3 + 1) / 3
        p, q = edges[edge_index]
        end = (p[0] + (q[0] - p[0]) * fraction,
               p[1] + (q[1] - p[1]) * fraction)
        draw.line([(p[0]*4, p[1]*4), (end[0]*4, end[1]*4)],
                  fill=(55, 112, 185, 255), width=8)
    elif i >= 17:
        progress = min(1, (i - 16) / 7)
        x, y = mid
        radius = 10
        draw.arc(((x-radius)*4, (y-radius)*4, (x+radius)*4, (y+radius)*4),
                 start=-90, end=-90 + 360*progress, fill=(230, 30, 38, 255), width=10)
    overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')

cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
       '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
       '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
for i in range(25):
    proc.stdin.write(np.asarray(frame(i)).tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait() != 0:
    raise RuntimeError(err.decode())
