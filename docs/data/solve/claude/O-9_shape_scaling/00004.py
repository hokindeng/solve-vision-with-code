"""Generate the analogy-completion video: scale trapezoid C by the A->B factor into the '?' slot."""
import numpy as np
from PIL import Image
import subprocess, os, shutil

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N = 16, 60

base = np.array(Image.open(SRC).convert('RGB')).astype(np.float32)
H, W, _ = base.shape

def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), xs.max(), ys.min(), ys.max()

r, g, b = base[..., 0], base[..., 1], base[..., 2]
red = (r > 150) & (g < 60)
# A: top-left, B: top-right, C: bottom-left
def quad_bbox(mask, x0, y0):
    x1, x2, y1, y2 = bbox(mask[y0:y0 + H // 2, x0:x0 + W // 2])
    return x1 + x0, x2 + x0, y1 + y0, y2 + y0
A = quad_bbox(red, 0, 0)
B = quad_bbox(red, W // 2, 0)
C = quad_bbox(red, 0, H // 2)
# scale factor from A -> B (average of width/height ratios)
sx = (B[1] - B[0]) / (A[1] - A[0])
sy = (B[3] - B[2]) / (A[3] - A[2])
FACTOR = (sx + sy) / 2
# '?' glyph in bottom-right quadrant: non-white, non-red pixels
nonwhite = base.min(axis=2) < 250
qmask = np.zeros((H, W), bool)
qx0 = W // 2 + 120  # exclude the arrow, which straddles the vertical midline
qmask[H // 2:, qx0:] = nonwhite[H // 2:, qx0:] & ~red[H // 2:, qx0:]
Q = bbox(qmask)
# target center: x from B's center, y from C's center
tcx = (B[0] + B[1]) / 2.0
tcy = (C[2] + C[3]) / 2.0

# Extract trapezoid sprite (RGBA, un-premultiplied against white background)
m = 4
x1, x2, y1, y2 = C[0] - m, C[1] + m, C[2] - m, C[3] + m
patch = base[y1:y2 + 1, x1:x2 + 1]
alpha = np.clip((255.0 - patch.min(axis=2)) / 255.0, 0, 1)
alpha = np.where(alpha > 0.02, alpha, 0)
col = np.zeros_like(patch)
nz = alpha > 0
col[nz] = np.clip((patch[nz] - (1 - alpha[nz])[:, None] * 255.0) / alpha[nz][:, None], 0, 255)
sprite = Image.fromarray(np.dstack([col, alpha * 255]).astype(np.uint8), 'RGBA')
sw, sh = sprite.size
scx, scy = (C[0] + C[1]) / 2.0 - x1, (C[2] + C[3]) / 2.0 - y1  # shape center inside sprite

# Question-mark erased image (white background there)
erased = base.copy()
qy1, qy2, qx1, qx2 = Q[2] - 3, Q[3] + 4, Q[0] - 3, Q[1] + 4
erased[qy1:qy2, qx1:qx2] = 255.0

def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

def render(i):
    t = i / (N - 1)
    q_alpha = 1.0 - smooth((i - 0) / 14.0)          # '?' fades out over frames 0..14
    s_alpha = smooth((i - 6) / 14.0)                 # trapezoid copy fades in frames 6..20
    scale = 1.0 + (FACTOR - 1.0) * smooth((i - 20) / 34.0)  # shrink frames 20..54
    img = erased * (1 - q_alpha) + base * q_alpha if q_alpha > 0 else erased.copy()
    if s_alpha > 0:
        nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
        sp = sprite.resize((nw, nh), Image.LANCZOS)
        sa = np.array(sp).astype(np.float32)
        px = tcx - scx * (nw / sw)
        py = tcy - scy * (nh / sh)
        ox, oy = int(round(px)), int(round(py))
        a = sa[..., 3:4] / 255.0 * s_alpha
        region = img[oy:oy + nh, ox:ox + nw]
        img[oy:oy + nh, ox:ox + nw] = region * (1 - a) + sa[..., :3] * a
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)

tmp = '/app/output/frames'
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
for i in range(N):
    fr = np.array(Image.open(SRC).convert('RGB')) if i == 0 else render(i)
    Image.fromarray(fr).save(f'{tmp}/{i:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-r', str(FPS), OUT], check=True)
shutil.rmtree(tmp, ignore_errors=True)
print(f'factor={FACTOR:.4f} target=({tcx:.1f},{tcy:.1f}) -> {OUT}')
