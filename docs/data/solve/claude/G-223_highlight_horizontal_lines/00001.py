import numpy as np, subprocess, math, os
from PIL import Image, ImageDraw
from scipy import ndimage

SRC = '/app/first_frame.png'; OUT = '/app/output/video.mp4'
W = H = 1024; FPS = 16; N = 48
base = np.array(Image.open(SRC).convert('RGB'))

# detect line segments as non-background connected components
mask = np.any(base != 255, axis=2)
lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
horiz = []
for sl in ndimage.find_objects(lab):
    h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
    if w > 3 * h and w > 20:  # horizontal line
        cy = (sl[0].start + sl[0].stop - 1) / 2; cx = (sl[1].start + sl[1].stop - 1) / 2
        horiz.append((cx, cy, w / 2 + 28))
print('horizontal lines:', horiz)

def ease(t): return t * t * (3 - 2 * t)

SS = 4
def frame(t):
    if not horiz: return base
    ov = Image.new('L', (W * SS, H * SS), 0); d = ImageDraw.Draw(ov)
    per = 1.0 / len(horiz)
    for i, (cx, cy, r) in enumerate(horiz):
        p = min(1, max(0, (t - i * per) / per)) if len(horiz) > 1 else t
        p = ease(p)
        if p <= 0: continue
        box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
        start = -100
        d.arc(box, start, start + 360 * p, fill=255, width=6 * SS)
    a = np.array(ov.resize((W, H), Image.LANCZOS)).astype(np.float32)[..., None] / 255
    out = base.astype(np.float32) * (1 - a) + 0 * a
    return out.round().astype(np.uint8)

os.makedirs('/app/output', exist_ok=True)
frames = [frame(0.0)] + [frame(k / (N - 4)) for k in range(1, N - 3)] + [frame(1.0)] * 3
assert len(frames) == N
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '16', OUT], stdin=subprocess.PIPE)
for f in frames: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('wrote', OUT)
