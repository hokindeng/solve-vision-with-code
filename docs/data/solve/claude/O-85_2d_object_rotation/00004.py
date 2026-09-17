import os, subprocess
import numpy as np, cv2
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS, ANGLE = 17, 16, 21.0  # clockwise degrees

img = np.array(Image.open(SRC).convert('RGB')).astype(np.uint8)
H, W = img.shape[:2]

# Foreground = non-white pixels; ignore the title band at the top.
fg = (img.min(axis=2) < 250).astype(np.uint8)
fg[:100] = 0
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 200]
ids = sorted(ids, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]

objects = []
bg = img.copy()
for i in ids:
    mask = (lab == i).astype(np.uint8)
    # Fill holes so the outline+interior form one solid shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(mask); cv2.drawContours(solid, cnts, -1, 1, -1)
    # dilate a bit to capture anti-aliased fringe
    grab = cv2.dilate(solid, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(solid)
    cx, cy = xs.mean(), ys.mean()
    layer = np.dstack([img, (grab * 255).astype(np.uint8)])  # RGBA
    # Make alpha soft at the edge: use distance to white for fringe pixels
    fringe = (grab == 1) & (solid == 0)
    a = (255 - img.min(axis=2)).astype(np.uint8)
    layer[..., 3][fringe] = a[fringe]
    objects.append((layer, (cx, cy)))
    bg[grab == 1] = 255

def render(t):
    frame = bg.astype(np.float32)
    ang = ANGLE * t
    for layer, (cx, cy) in objects:
        # positive angle in cv2 = CCW in image coords with y-down => use -ang for clockwise
        M = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
        rot = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))
        alpha = rot[..., 3:4].astype(np.float32) / 255.0
        frame = frame * (1 - alpha) + rot[..., :3].astype(np.float32) * alpha
    return np.clip(frame + 0.5, 0, 255).astype(np.uint8)

os.makedirs('/app/output', exist_ok=True)
tmp = '/app/output/frames'; os.makedirs(tmp, exist_ok=True)
for k in range(N_FRAMES):
    t = k / (N_FRAMES - 1)
    fr = img if k == 0 else render(t)
    Image.fromarray(fr).save(f'{tmp}/{k:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-r', str(FPS), OUT], check=True)
print('wrote', OUT)
