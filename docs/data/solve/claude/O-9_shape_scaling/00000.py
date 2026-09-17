import math, subprocess, os
import numpy as np
from PIL import Image, ImageDraw

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 60, 16

# Measured from first_frame.png
S = 0.865                  # scale factor inferred from heart A -> heart B
C_CENTER = (255.5, 768.5)  # pentagon C center (exact redraw match)
R = 75.0                   # pentagon circumradius
DX = 514.0                 # column offset A -> B (and C -> ?)
Q_BOX = (740, 735, 800, 800)  # region containing the '?' glyph
FILL, OUTLINE = (255, 255, 0), (0, 0, 0)

base = Image.open(SRC).convert('RGB')
base_np = np.array(base)

# Background with the '?' removed (white there; the glyph sits on a plain white cell)
no_q = base_np.copy()
x0, y0, x1, y1 = Q_BOX
no_q[y0:y1, x0:x1] = 255

def pentagon_pts(cx, cy, r, ss=1):
    return [((cx + r * math.sin(2 * math.pi * k / 5)) * ss,
             (cy - r * math.cos(2 * math.pi * k / 5)) * ss) for k in range(5)]

def draw_pentagon(img_np, cx, cy, r, alpha=1.0, aa=4):
    """Composite a pentagon (yellow fill, 1px black outline) onto img_np."""
    if aa == 1:
        im = Image.fromarray(img_np.copy())
        ImageDraw.Draw(im).polygon(pentagon_pts(cx, cy, r), fill=FILL, outline=OUTLINE)
        return np.array(im)
    # supersampled layer for smooth intermediate frames
    W, H = img_np.shape[1], img_np.shape[0]
    layer = Image.new('RGBA', (W * aa, H * aa), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon(pentagon_pts(cx, cy, r, aa), fill=FILL + (255,), outline=OUTLINE + (255,), width=aa)
    layer = layer.resize((W, H), Image.LANCZOS)
    la = np.array(layer).astype(np.float32)
    a = (la[:, :, 3:4] / 255.0) * alpha
    out = img_np.astype(np.float32) * (1 - a) + la[:, :, :3] * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)

os.makedirs('/app/output', exist_ok=True)
tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)

tx, ty = C_CENTER[0] + DX, C_CENTER[1]
FADE_END = 14  # frames 0..FADE_END: '?' fades out, full-size copy of C fades in at '?'
for i in range(N_FRAMES):
    if i == 0:
        frame = base_np
    elif i <= FADE_END:
        t = ease(i / FADE_END)
        bg = (base_np.astype(np.float32) * (1 - t) + no_q.astype(np.float32) * t + 0.5).astype(np.uint8)
        frame = draw_pentagon(bg, tx, ty, R, alpha=t)
    else:
        t = ease((i - FADE_END) / (N_FRAMES - 1 - FADE_END))
        r = R * (1 - t) + R * S * t
        frame = draw_pentagon(no_q, tx, ty, r, aa=1 if i == N_FRAMES - 1 else 4)
    Image.fromarray(frame).save(f'{tmp}/f{i:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/f%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '15', '-r', str(FPS), OUT], check=True)
print('wrote', OUT)
