import numpy as np, subprocess, os
from PIL import Image, ImageDraw

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
NF, FPS = 10, 16

base = Image.open(SRC).convert('RGB')
arr = np.array(base)
ink = (arr < 128).all(axis=2)
H, W = ink.shape

# Trace the wave: mean row index of ink per column -> wave value = -y
cols = np.nonzero(ink.any(axis=0))[0]
ys = np.array([np.nonzero(ink[:, c])[0].mean() for c in cols])
val = -ys
# Smooth slightly to remove rasterization jitter
k = 5
vs = np.convolve(np.pad(val, k // 2, mode='edge'), np.ones(k) / k, mode='valid')
# Local maxima (strict vs both neighbors); merge plateaus of near-equal values
peaks = []
for i in range(1, len(vs) - 1):
    if vs[i] > vs[i - 1] and vs[i] >= vs[i + 1]:
        peaks.append(i)
# Keep prominent maxima only (reject rasterization noise), merge close ones
amp = vs.max() - vs.min()
final = []
for i in peaks:
    lo = max(0, i - 40); hi = min(len(vs), i + 41)
    if vs[i] >= vs[lo:hi].max() and (vs[i] - min(vs[lo], vs[hi - 1])) > 0.02 * amp:
        if not final or cols[i] - final[-1][0] > 20:
            # refine: use topmost ink pixel in the neighbourhood
            seg = slice(max(0, i - 6), min(len(cols), i + 7))
            j = np.argmin(ys[seg]) + seg.start
            cx = float(cols[j])
            cy = float(np.nonzero(ink[:, int(cx)])[0].min()) + 1.0
            final.append((cx, cy))
peaks_xy = sorted(final)
print('peaks:', peaks_xy)

R, LW, DOT = 22, 4, 5
RED = (255, 0, 0)

def frame(t):
    """t in [0,1]: progress; peaks appear one by one, outline grows then dot."""
    im = base.copy()
    d = ImageDraw.Draw(im)
    n = len(peaks_xy)
    for idx, (cx, cy) in enumerate(peaks_xy):
        p = t * n - idx  # per-peak progress
        if p <= 0:
            continue
        p = min(p, 1.0)
        # phase 1: outline arc sweeps around; phase 2: dot appears
        sweep = min(p / 0.7, 1.0)
        bbox = [cx - R, cy - R, cx + R, cy + R]
        if sweep >= 1.0:
            d.ellipse(bbox, outline=RED, width=LW)
        else:
            d.arc(bbox, start=-90, end=-90 + 360 * sweep, fill=RED, width=LW)
        if p >= 0.7:
            d.ellipse([cx - DOT, cy - DOT, cx + DOT, cy + DOT], fill=RED)
    return im

os.makedirs('/app/output', exist_ok=True)
tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
for i in range(NF):
    t = i / (NF - 1)
    frame(t).save(f'{tmp}/f{i:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/f%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', OUT], check=True)
print('wrote', OUT)
