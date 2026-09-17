from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Extract the original character, retaining its exact outline and mouth pixels.
green = original[20,20].copy()
mask = np.any(original[3:254,3:254] != green, axis=2)
yy, xx = np.where(mask)
x0, x1 = xx.min()+3, xx.max()+4
y0, y1 = yy.min()+3, yy.max()+4
sprite = original[y0:y1,x0:x1].copy()
alpha = np.any(sprite != green, axis=2)
background = original.copy()
region = background[y0:y1,x0:x1]
region[alpha] = green

# Without a no-revisit constraint, positive cycles make the objective unbounded.
# Exhaustively optimize over simple orthogonal paths to the goal.
cost = [0,10,50,30,10,50,50,10,50,40,30,40,30,30,10,30]
adj = []
for v in range(16):
    r,c = divmod(v,4)
    adj.append([rr*4+cc for rr,cc in [(r,c+1),(r+1,c),(r,c-1),(r-1,c)]
                if 0 <= rr < 4 and 0 <= cc < 4])
best_score = -1
best_path = None
def visit(v, seen, score, path):
    global best_score, best_path
    if v == 15:
        if score > best_score:
            best_score, best_path = score, path.copy()
        return
    for nxt in adj[v]:
        if not seen & (1 << nxt):
            visit(nxt, seen | (1 << nxt), score + cost[nxt], path+[nxt])
visit(0,1,0,[0])
print('Optimal simple path:', best_path, 'score:',best_score)

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
       '-preset','slow','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for frame_index in range(91):
    # Hold the initial and final poses briefly, with uniform motion between cells.
    progress = min(1., max(0., (frame_index-2)/85))*(len(best_path)-1)
    step = min(int(progress),len(best_path)-2)
    t = progress-step
    a,b = best_path[step:step+2]
    ar,ac = divmod(a,4)
    br,bc = divmod(b,4)
    dx = round(256*(ac+(bc-ac)*t))
    dy = round(256*(ar+(br-ar)*t))
    frame = background.copy()
    dest = frame[y0+dy:y1+dy,x0+dx:x1+dx]
    dest[alpha] = sprite[alpha]
    if frame_index == 0:
        assert np.array_equal(frame,original)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
