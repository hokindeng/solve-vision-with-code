"""Outline the topmost (unobscured) triangle with a red outline, drawn progressively."""
import numpy as np, cv2, subprocess, os
from PIL import Image

BASE = np.array(Image.open('/app/first_frame.png').convert('RGB'))
H, W = BASE.shape[:2]
N_FRAMES, FPS = 40, 16
OUT_DIR = '/app/output'
os.makedirs(OUT_DIR, exist_ok=True)


def find_topmost():
    """The topmost shape is the one whose colour region is a full, unbroken convex polygon
    (its contour area equals its convex-hull area, i.e. nothing cuts into it)."""
    cols, counts = np.unique(BASE.reshape(-1, 3), axis=0, return_counts=True)
    best = None
    for c, n in zip(cols, counts):
        if n < 2000 or (c == 255).all():
            continue
        m = (np.abs(BASE.astype(int) - c).sum(2) < 30).astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.contourArea(cv2.convexHull(cnt))
        ratio = area / max(hull, 1)
        poly = cv2.approxPolyDP(cnt, 3, True).reshape(-1, 2)
        if best is None or ratio > best[0]:
            best = (ratio, poly)
    return best[1].astype(float)


def outline_pixels(poly, frac, thickness=6):
    """Points along the polygon perimeter up to `frac` of its length."""
    pts = list(poly) + [poly[0]]
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    lens = [np.linalg.norm(b - a) for a, b in segs]
    total = sum(lens)
    remain = frac * total
    path = [pts[0]]
    for (a, b), L in zip(segs, lens):
        if remain <= 0:
            break
        if remain >= L:
            path.append(b); remain -= L
        else:
            path.append(a + (b - a) * (remain / L)); remain = 0
    return np.array(path)


def render(frame_idx):
    img = BASE.copy()
    if frame_idx == 0:
        return img
    frac = frame_idx / (N_FRAMES - 1)
    frac = frac * frac * (3 - 2 * frac)  # ease in-out
    path = outline_pixels(POLY, frac)
    thick = 6
    overlay = img.copy()
    pts = np.round(path).astype(np.int32).reshape(-1, 1, 2)
    closed = frame_idx == N_FRAMES - 1
    cv2.polylines(overlay, [pts], closed, (255, 0, 0), thick, cv2.LINE_AA)
    return overlay


POLY = find_topmost()
frames = [render(i) for i in range(N_FRAMES)]

raw = os.path.join(OUT_DIR, 'frames.rgb')
with open(raw, 'wb') as f:
    for fr in frames:
        f.write(np.ascontiguousarray(fr).tobytes())
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{W}x{H}', '-r', str(FPS), '-i', raw,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15',
                os.path.join(OUT_DIR, 'video.mp4')], check=True)
os.remove(raw)
Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, 'last_frame.png'))
print('vertices:', POLY.tolist())
