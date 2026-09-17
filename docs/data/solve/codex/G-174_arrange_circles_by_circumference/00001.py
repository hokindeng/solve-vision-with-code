from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = cv2.imread(str(ROOT / 'first_frame.png'))
mask = np.any(source != 255, axis=2).astype(np.uint8)
n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
# Keep the original raster pixels of each circle, including its black outline.
order = sorted(range(1, n), key=lambda i: stats[i, 2], reverse=True)
gap = 32
width = sum(stats[i, 2] for i in order) + gap * (len(order)-1)
x = (1024-width)//2
objects = []
controls = [(230,180), (620,730), (260,440), (750,790), (850,110)]
for i, control in zip(order, controls):
    sx, sy, w, h, _ = stats[i]
    pixels = source[sy:sy+h, sx:sx+w].copy()
    local_mask = labels[sy:sy+h, sx:sx+w] == i
    target = np.array([x+(w-1)/2, 512.0])
    objects.append((pixels, local_mask, centers[i], np.array(control), target))
    x += w+gap

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
       '-crf','15','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for frame in range(80):
    t = frame/79
    u = t*t*t*(10 + t*(-15+6*t))
    canvas = np.full_like(source,255)
    for pixels, local_mask, start, control, target in objects:
        pos = (1-u)**2*start + 2*(1-u)*u*control + u*u*target
        h,w = local_mask.shape
        px,py = np.rint(pos-np.array([(w-1)/2,(h-1)/2])).astype(int)
        region = canvas[py:py+h,px:px+w]
        region[local_mask] = pixels[local_mask]
    if frame == 0:
        assert np.array_equal(canvas, source)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
