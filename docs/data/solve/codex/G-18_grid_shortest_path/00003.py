from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
mask = (original[:,:,0] == 255) & (original[:,:,1] == 0) & (original[:,:,2] == 0)
y, x = np.where(mask)
background = original.copy()
background[mask] = (255, 165, 0)
# Move right four cells, then up one: Manhattan distance is five cells.
waypoints = [(0,0), (102,0), (204,0), (306,0), (408,0), (408,-102)]
command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',
           str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for index in range(45):
    progress = min(index / 42 * 5, 5)
    segment = min(int(progress), 4)
    t = progress - segment
    a, b = waypoints[segment], waypoints[segment+1]
    dx = round(a[0] + (b[0]-a[0])*t)
    dy = round(a[1] + (b[1]-a[1])*t)
    frame = background.copy()
    frame[y+dy, x+dx] = original[y,x]
    if index == 0:
        assert np.array_equal(frame, original)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
errors = proc.stderr.read()
if proc.wait():
    raise RuntimeError(errors.decode())
print(OUT / 'video.mp4')
