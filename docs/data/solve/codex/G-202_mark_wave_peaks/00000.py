from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
# The curve's visual crests are minima of its image-space y coordinate.
mask = a.min(axis=2) < 100
ys = np.full(a.shape[1], np.nan)
for x in range(a.shape[1]):
    rows = np.flatnonzero(mask[:, x])
    if len(rows):
        ys[x] = rows.mean()
candidates = []
for x in range(25, len(ys)-25):
    if np.isfinite(ys[x]) and ys[x] == np.nanmin(ys[x-25:x+26]):
        candidates.append(x)
groups = []
for x in candidates:
    if not groups or x-groups[-1][-1] > 25:
        groups.append([x])
    else:
        groups[-1].append(x)
peaks = [(int(round(np.mean(g))), int(round(np.nanmin(ys[g])))) for g in groups]
assert len(peaks) == 3, peaks

scale = 4
frames = []
for index in range(10):
    overlay = Image.new('RGBA', (base.width*scale, base.height*scale))
    draw = ImageDraw.Draw(overlay)
    for p, (x,y) in enumerate(peaks):
        progress = min(1.0, max(0.0, (index-3*p)/3))
        if progress == 0:
            continue
        radius = 24
        box = tuple(v*scale for v in (x-radius,y-radius,x+radius,y+radius))
        draw.arc(box, start=-90, end=-90+360*progress, fill=(235,0,0,255), width=3*scale)
        dot = 4
        draw.ellipse(tuple(v*scale for v in (x-dot,y-dot,x+dot,y+dot)), fill=(235,0,0,255))
    overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
    frame = Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')
    frames.append(frame)
# Original raster is used unchanged for frame zero; overlays affect only annotations.
assert np.array_equal(np.asarray(frames[0]), a)
proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for frame in frames:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
print('Created', OUT/'video.mp4', 'with peaks', peaks)
