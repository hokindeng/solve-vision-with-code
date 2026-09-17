from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
start, goal = (7, 0), (1, 6)
obstacles = {(0,5),(0,8),(2,4),(2,9),(3,3),(3,9),(4,3),(6,5),(7,3),(8,3),(9,1),(9,9)}
queue = deque([start])
parent = {start: None}
while queue:
    cell = queue.popleft()
    if cell == goal:
        break
    for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
        nxt = (cell[0]+dr, cell[1]+dc)
        if 0 <= nxt[0] < 10 and 0 <= nxt[1] < 10 and nxt not in obstacles and nxt not in parent:
            parent[nxt] = cell
            queue.append(nxt)
path = []
cell = goal
while cell is not None:
    path.append(cell)
    cell = parent[cell]
path.reverse()
assert len(path) == 13
# Reuse the original agent pixels, including its darker circular outline.
mask = (original[:,:,0] >= 190) & (original[:,:,1] >= 140) & (original[:,:,1] <= 205) & (original[:,:,2] == 0)
y, x = np.nonzero(mask)
sprite = original[y, x].copy()
background = original.copy()
background[y, x] = (0,100,255)
proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','medium','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for frame_index in range(58):
    if frame_index == 0:
        frame = original
    else:
        progress = frame_index / 57 * (len(path)-1)
        segment = min(int(progress),len(path)-2)
        fraction = progress-segment
        a, b = path[segment], path[segment+1]
        row = a[0]+(b[0]-a[0])*fraction
        col = a[1]+(b[1]-a[1])*fraction
        dx = round((col-start[1])*93)
        dy = round((row-start[0])*93)
        frame = background.copy()
        frame[y+dy,x+dx] = sprite
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Video encoding failed')
