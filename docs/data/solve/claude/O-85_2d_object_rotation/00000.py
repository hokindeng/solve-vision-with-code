import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS, ANGLE = 17, 16, 176.0

img = np.array(Image.open(SRC).convert('RGB'))
H, W = img.shape[:2]
bg_color = np.array([255, 255, 255], np.uint8)

# Segment non-background pixels and pick the 3 largest connected components (the objects).
nonbg = (np.abs(img.astype(int) - bg_color).sum(2) > 20).astype(np.uint8)
nonbg = cv2.morphologyEx(nonbg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(nonbg, connectivity=8)
order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:3]

objects = []
background = img.copy()
for i in order:
    mask = (lab == i)
    # fill interior holes so the whole object (outline + fill) is treated as one body
    m8 = mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m8); cv2.drawContours(filled, cnts, -1, 255, -1)
    ys, xs = np.nonzero(filled)
    cx, cy = xs.mean(), ys.mean()          # centroid of the object
    rgba = np.dstack([img, filled])
    objects.append((rgba, cx, cy))
    background[filled > 0] = bg_color        # remove object from static background

def render(t):
    frame = background.copy()
    theta = ANGLE * t
    for rgba, cx, cy in objects:
        # positive angle in cv2.getRotationMatrix2D = counterclockwise on screen
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        rot = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = rot[..., 3:4].astype(np.float32) / 255.0
        frame = (rot[..., :3].astype(np.float32) * a + frame.astype(np.float32) * (1 - a)).round().astype(np.uint8)
    return frame

frames = [img.copy()] + [render(k / (N_FRAMES - 1)) for k in range(1, N_FRAMES)]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '12', '-preset', 'slow', OUT], stdin=subprocess.PIPE)
for f in frames: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('wrote', OUT, 'frames', len(frames), 'centroids', [(round(c[1]), round(c[2])) for c in objects])
