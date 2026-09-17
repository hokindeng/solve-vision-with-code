from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(base)
N = 15
centers = [int((i + .5) * 1024 / N) for i in range(N)]
walk = [[max(a[centers[y], centers[x]]) > 100 for x in range(N)] for y in range(N)]
start, end = (1, 2), (14, 12)
walk[end[1]][end[0]] = True
queue = deque([start])
prev = {start: None}
while queue:
    x, y = queue.popleft()
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        v = x + dx, y + dy
        if 0 <= v[0] < N and 0 <= v[1] < N and walk[v[1]][v[0]] and v not in prev:
            prev[v] = (x, y)
            queue.append(v)
assert end in prev
route = []
v = end
while v is not None:
    route.append(v)
    v = prev[v]
route.reverse()
points = [(centers[x], centers[y]) for x, y in route]
# Only the white corridor pixels may be painted; keep all existing symbols intact.
white = np.all(a == 255, axis=2)
encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
    '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')], stdin=subprocess.PIPE)
for frame in range(87):
    result = a.copy()
    if frame:
        progress = (len(points) - 1) * frame / 86
        idx = min(int(progress), len(points) - 2)
        t = progress - idx
        x = points[idx][0] + t * (points[idx+1][0] - points[idx][0])
        y = points[idx][1] + t * (points[idx+1][1] - points[idx][1])
        layer = base.copy()
        draw = ImageDraw.Draw(layer)
        traveled = points[:idx+1] + [(round(x), round(y))]
        draw.line(traveled, fill=(44, 198, 45), width=14, joint='curve')
        for px, py in points[:idx+1]:
            draw.ellipse((px-7, py-7, px+7, py+7), fill=(44,198,45))
        draw.ellipse((x-20,y-20,x+20,y+20), fill=(44,198,45))
        result[white] = np.array(layer)[white]
    encoder.stdin.write(result.tobytes())
encoder.stdin.close()
assert encoder.wait() == 0
print(f'Created {OUT / "video.mp4"}: 87 frames, {len(route)-1} maze steps')
