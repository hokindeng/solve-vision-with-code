import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import subprocess, os, shutil

SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
FRAMES_DIR = os.path.join(OUT_DIR, 'frames')
N_FRAMES = 100
FPS = 16

base = Image.open(SRC).convert('RGB')
arr = np.array(base)

# --- detect dots: connected components of non-white pixels ---
nonwhite = (arr.astype(int).sum(axis=2) < 3 * 250)
nonwhite = ndimage.binary_closing(nonwhite, iterations=2)
nonwhite = ndimage.binary_fill_holes(nonwhite)
lab, n = ndimage.label(nonwhite)
comps = []
for i in range(1, n + 1):
    m = lab == i
    if m.sum() < 500:
        continue
    ys, xs = np.nonzero(m)
    comps.append(((xs.mean(), ys.mean()), m))

# Number labels read from the frame (approximate centers), used to order the detected dots.
approx = {1: (888, 153), 2: (424, 573), 3: (171, 390), 4: (782, 648),
          5: (652, 819), 6: (630, 228), 7: (315, 187)}
ordered = []
dot_mask = np.zeros(arr.shape[:2], bool)
for k in sorted(approx):
    ax, ay = approx[k]
    (c, m) = min(comps, key=lambda cm: (cm[0][0] - ax) ** 2 + (cm[0][1] - ay) ** 2)
    ordered.append(c)
    dot_mask |= ndimage.binary_dilation(m, iterations=1)
assert len(ordered) == 7

# --- animation schedule ---
segments = len(ordered) - 1            # 6 lines
draw_frames = 90                        # frames 1..90 animate, rest hold
per_seg = draw_frames / segments
LINE_W = 5
RED = (255, 0, 0)

def render(t_frame):
    if t_frame == 0:
        return base.copy()
    im = base.copy()
    d = ImageDraw.Draw(im)
    progress = min(t_frame / per_seg, segments)  # in units of segments
    for s in range(segments):
        p0 = ordered[s]; p1 = ordered[s + 1]
        f = min(max(progress - s, 0.0), 1.0)
        if f <= 0:
            break
        x = p0[0] + (p1[0] - p0[0]) * f
        y = p0[1] + (p1[1] - p0[1]) * f
        d.line([p0, (x, y)], fill=RED, width=LINE_W)
    out = np.array(im)
    out[dot_mask] = arr[dot_mask]       # keep dots and their labels untouched
    return Image.fromarray(out)

shutil.rmtree(FRAMES_DIR, ignore_errors=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
for i in range(N_FRAMES):
    render(i).save(os.path.join(FRAMES_DIR, f'{i:04d}.png'))

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', os.path.join(FRAMES_DIR, '%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15',
                os.path.join(OUT_DIR, 'video.mp4')], check=True)
shutil.rmtree(FRAMES_DIR, ignore_errors=True)
print('done', ordered)
