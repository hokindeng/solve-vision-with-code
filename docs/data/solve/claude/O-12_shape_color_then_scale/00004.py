"""Generate the analogy-completion video: bottom-row oval gets color change then size change."""
import numpy as np, subprocess, os
from PIL import Image, ImageDraw

BASE = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
W = H = 1024; FPS = 16; N = 60
OLIVE = (153, 132, 30); TEAL = (66, 191, 149); OUTLINE = (0, 0, 0)
ROW_Y = 682; CX2, CX3 = 518, 854            # centres of the two '?' cells
RX, RY = 98.0, 49.0                          # large oval half-axes (from column D)
Q1 = (500, 655, 537, 709); Q2 = (836, 655, 873, 709)  # '?' glyph boxes (x0,y0,x1,y1)
SS = 4                                        # supersampling

base = Image.open(BASE).convert('RGB')
base_np = np.array(base).astype(np.float32)
white = np.full_like(base_np, 255.0)

def smooth(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)

def lerp_c(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def oval_layer(cx, cy, rx, ry, fill, alpha):
    """RGBA layer with an anti-aliased outlined ellipse."""
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS],
              fill=fill + (255,), outline=OUTLINE + (255,), width=2 * SS)
    lay = big.resize((W, H), Image.LANCZOS)
    a = np.array(lay).astype(np.float32)
    a[..., 3] *= alpha
    return a

def composite(dst, lay):
    a = lay[..., 3:4] / 255.0
    return dst * (1 - a) + lay[..., :3] * a

def erase_q(img, box, k):
    """Fade the question mark in box towards white by factor k (1 = fully gone)."""
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = base_np[y0:y1, x0:x1] * (1 - k) + white[y0:y1, x0:x1] * k
    return img

frames = []
for f in range(N):
    img = base_np.copy()
    # ---- stage 1: column 2 -> large oval appears, then colour olive -> teal
    a1 = smooth((f - 4) / 10.0)          # frames 4..14 fade in
    c1 = smooth((f - 14) / 14.0)         # frames 14..28 recolour
    if a1 > 0:
        img = erase_q(img, Q1, a1)
        img = composite(img, oval_layer(CX2, ROW_Y, RX, RY, lerp_c(OLIVE, TEAL, c1), a1))
    # ---- stage 2: column 3 -> large teal oval appears, then shrinks to small
    a2 = smooth((f - 30) / 10.0)         # frames 30..40 fade in
    s2 = smooth((f - 40) / 14.0)         # frames 40..54 shrink
    if a2 > 0:
        img = erase_q(img, Q2, a2)
        k = 1.0 - 0.5 * s2
        img = composite(img, oval_layer(CX3, ROW_Y, RX * k, RY * k, TEAL, a2))
    frames.append(np.clip(np.round(img), 0, 255).astype(np.uint8))

tmp = '/app/output/_frames'; os.makedirs(tmp, exist_ok=True)
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(f'{tmp}/f{i:03d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/f%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-r', str(FPS), OUT], check=True)
for p in os.listdir(tmp): os.remove(os.path.join(tmp, p))
os.rmdir(tmp)
print('wrote', OUT)
