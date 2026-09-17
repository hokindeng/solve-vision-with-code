from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The given square widths are 43, 76, 55, 43, 76: next is 55.
color = base[500, 430].copy()
x0, y0, side = 857, 485, 55
# Trace each edge in turn, then fill the medium square from top to bottom.
perimeter = ([(x0+x, y0) for x in range(side)] +
             [(x0+side-1, y0+y) for y in range(1, side)] +
             [(x0+x, y0+side-1) for x in range(side-2, -1, -1)] +
             [(x0, y0+y) for y in range(side-2, 0, -1)])
cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
       '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
       '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(60):
    frame = base.copy()
    if i >= 8:
        count = min(len(perimeter), int(len(perimeter) * (i-7) / 28))
        for x, y in perimeter[:count]:
            frame[y, x] = color
    if i >= 36:
        rows = min(side, int(side * (i-35) / 20))
        frame[y0:y0+rows, x0:x0+side] = color
    # All changes are confined to the final square's footprint.
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait():
    raise RuntimeError(err.decode())
