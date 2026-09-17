from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
heart_mask = (source[:,:,0] > 240) & (source[:,:,1] > 80) & (source[:,:,1] < 160) & (source[:,:,2] < 110)
blue_mask = (source[:,:,2] > 240) & (source[:,:,0] < 20) & (source[:,:,1] < 20)
hy, hx = np.where(heart_mask)
by, bx = np.where(blue_mask)
base = source.copy()
base[hy, hx] = 255

def ease(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'slow',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(32):
    frame = base.copy()
    shift = int(round(105 * ease(i/13)))
    frame[hy, hx + shift] = source[hy, hx]
    if i >= 14:
        alpha = ease((i-14)/6)
        center_y = 397 + int(round(115 * ease((i-20)/11)))
        dx = 617 - 946
        dy = center_y - 77
        yy, xx = by + dy, bx + dx
        frame[yy,xx] = np.round(frame[yy,xx]*(1-alpha) + source[by,bx]*alpha).astype(np.uint8)
    if i == 0:
        assert np.array_equal(frame, source)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait() != 0:
    raise RuntimeError(err.decode())
