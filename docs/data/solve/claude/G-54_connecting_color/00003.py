"""Generate a video in which, for each color, a smooth curve is drawn step by
step connecting the leftmost and rightmost shapes of that color."""
import numpy as np, cv2, subprocess, os, tempfile, shutil
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "first_frame.png")
OUT = os.path.join(ROOT, "output", "video.mp4")
W = H = 1024
FPS, N_FRAMES = 16, 48
THICK = 6

base = np.array(Image.open(SRC).convert("RGB"))
bg = base[0, 0].copy()

# ---- detect shapes grouped by exact color ------------------------------------
nonbg = np.any(base != bg, axis=2)
colors = np.unique(base[nonbg].reshape(-1, 3), axis=0)
shapes = {}  # color -> list of (centroid, mask)
for col in colors:
    m = np.all(base == col, axis=2).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    lst = [(cent[i], lab == i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 200]
    if len(lst) >= 2:
        shapes[tuple(int(c) for c in col)] = lst
all_shape_mask = nonbg.copy()
# order colors by vertical position (top to bottom) for a natural drawing order
order = sorted(shapes, key=lambda c: np.mean([s[0][1] for s in shapes[c]]))

# ---- build curves ------------------------------------------------------------
def bezier(p0, p1, p2, n=400):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def curve_mask(pts, thick):
    m = np.zeros((H, W), np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 255, thick, cv2.LINE_AA)
    return m

def clip_to_bg(pts, ma, mb):
    """keep the part of the polyline between leaving shape A and entering shape B"""
    inside_a = [ma[int(round(y)) % H, int(round(x)) % W] for x, y in pts]
    inside_b = [mb[int(round(y)) % H, int(round(x)) % W] for x, y in pts]
    i0 = max(i for i, v in enumerate(inside_a) if v) if any(inside_a) else 0
    i1 = min(i for i, v in enumerate(inside_b) if v) if any(inside_b) else len(pts) - 1
    return pts[i0:i1 + 1]

curves = []          # list of (color, clipped points)
occupied = np.zeros((H, W), np.uint8)  # previously drawn curves (dilated)
for col in order:
    lst = sorted(shapes[col], key=lambda s: s[0][0])
    (ca, ma), (cb, mb) = lst[0], lst[-1]
    others = all_shape_mask & ~ma & ~mb
    p0, p2 = np.array(ca, float), np.array(cb, float)
    mid = (p0 + p2) / 2
    d = p2 - p0
    perp = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    best = None
    for off in [60, -60, 120, -120, 200, -200, 300, -300, 0]:
        pts = clip_to_bg(bezier(p0, mid + perp * off, p2), ma, mb)
        cm = curve_mask(pts, THICK + 4) > 0
        if not (cm & (others | (occupied > 0))).any() and pts[:, 0].min() > 4 and pts[:, 0].max() < W - 4 \
                and pts[:, 1].min() > 4 and pts[:, 1].max() < H - 4:
            best = pts
            break
    if best is None:
        best = clip_to_bg(bezier(p0, mid, p2), ma, mb)
    curves.append((col, best))
    occupied |= curve_mask(best, THICK + 6)

# ---- render frames -----------------------------------------------------------
draw_ok = ~all_shape_mask  # only ever alter background pixels
per = (N_FRAMES - 3) // len(curves)  # frames per curve; frame 0 untouched, tail holds
frames = []
for f in range(N_FRAMES):
    img = base.copy()
    for k, (col, pts) in enumerate(curves):
        start = 1 + k * per
        prog = np.clip((f - start + 1) / per, 0, 1)
        if prog <= 0:
            continue
        n = max(2, int(round(prog * len(pts))))
        layer = img.copy()
        cv2.polylines(layer, [np.round(pts[:n]).astype(np.int32)], False, col, THICK, cv2.LINE_AA)
        img[draw_ok] = layer[draw_ok]
    frames.append(img)

tmp = tempfile.mkdtemp()
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "12", "-r", str(FPS), OUT], check=True)
shutil.rmtree(tmp)
print("wrote", OUT, len(frames), "frames;", [(c, len(p)) for c, p in curves])
