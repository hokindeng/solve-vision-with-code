from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Extract precisely the original Pac-Man pixels and restore the solid start cell.
green = original[10, 10].copy()
mask = np.zeros(original.shape[:2], dtype=bool)
mask[20:240, 20:240] = np.any(original[20:240, 20:240] != green, axis=2)
ys, xs = np.where(mask)
colors = original[ys, xs].copy()
background = original.copy()
background[ys, xs] = green

# With positive costs, unrestricted repeated visits have no finite maximum.
# Use the finite simple-path interpretation: each cell may be visited once.
costs = [0,40,50,50,10,10,30,40,40,50,30,10,10,10,40,20]
adj = []
for p in range(16):
    r,c = divmod(p,4)
    adj.append([rr*4+cc for rr,cc in [(r,c+1),(r+1,c),(r,c-1),(r-1,c)]
                if 0 <= rr < 4 and 0 <= cc < 4])
best_score = -1
best_path = None
def search(p, visited, score, path):
    global best_score, best_path
    if p == 15:
        if score > best_score:
            best_score, best_path = score, path[:]
        return
    for q in adj[p]:
        if not visited & (1 << q):
            search(q, visited | (1 << q), score + costs[q], path + [q])
search(0,1,0,[0])
print('Maximum simple-path cost:', best_score, 'Path:', best_path)

frames = 91
command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
           '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
           '-an','-c:v','libx264','-crf','18','-preset','slow','-pix_fmt','yuv420p',
           '-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for i in range(frames):
    if i == 0:
        frame = original
    else:
        # Small pauses at each cell make individual orthogonal steps legible.
        progress = np.clip((i-3)/84,0,1) * (len(best_path)-1)
        segment = min(int(progress),len(best_path)-2)
        fraction = np.clip((progress-segment)/0.80,0,1)
        fraction = fraction*fraction*(3-2*fraction)
        a,b = best_path[segment:segment+2]
        ar,ac = divmod(a,4)
        br,bc = divmod(b,4)
        dx = round(256*(ac+(bc-ac)*fraction))
        dy = round(256*(ar+(br-ar)*fraction))
        frame = background.copy()
        frame[ys+dy,xs+dx] = colors
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
