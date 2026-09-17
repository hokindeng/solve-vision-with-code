import numpy as np
from PIL import Image, ImageDraw
import subprocess, os, math

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS, SS = 48, 16, 4  # SS = supersampling factor

base = Image.open(SRC).convert("RGB")
arr = np.array(base).astype(int)
W, H = base.size

# --- detect line segments as connected components of non-background pixels ---
bg = arr[0, 0]
mask = (np.abs(arr - bg).sum(axis=2) > 40)
try:
    import cv2
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] > 30]
except ImportError:
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    comps = []
    for sl in ndimage.find_objects(lab):
        comps.append((sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start))

horizontal = [(x, y, w, h) for (x, y, w, h) in comps if w > 3 * h]
horizontal.sort(key=lambda c: (c[1], c[0]))  # top-to-bottom order

circles = []  # (cx, cy, r)
for x, y, w, h in horizontal:
    cx, cy = x + w / 2, y + h / 2
    r = w / 2 + 22
    circles.append((cx, cy, r))

STROKE = 6

def render(frame_idx):
    """Return frame image. Circles are drawn sequentially as growing arcs."""
    t = frame_idx / (N_FRAMES - 1)  # 0..1
    layer = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(layer)
    k = len(circles)
    for i, (cx, cy, r) in enumerate(circles):
        # each circle gets an equal share of the timeline, starting slightly after t=0
        t0, t1 = 0.04 + i * (0.96 / k), 0.04 + (i + 1) * (0.96 / k)
        p = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        if p <= 0:
            continue
        p = p * p * (3 - 2 * p)  # ease in/out
        sweep = 360 * p
        box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
        start = -90
        if p >= 1.0:
            d.ellipse(box, outline=255, width=STROKE * SS)
        else:
            d.arc(box, start=start, end=start + sweep, fill=255, width=STROKE * SS)
            # round caps
            for ang in (start, start + sweep):
                a = math.radians(ang)
                px, py = (cx + r_mid(r) * math.cos(a)) * SS, (cy + r_mid(r) * math.sin(a)) * SS
                rr = STROKE * SS / 2
                d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=255)
    alpha = layer.resize((W, H), Image.LANCZOS)
    frame = base.copy()
    black = Image.new("RGB", (W, H), (0, 0, 0))
    frame.paste(black, (0, 0), alpha)
    return frame

def r_mid(r):
    return r - STROKE / 2

os.makedirs(OUT_DIR, exist_ok=True)
frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)
for i in range(N_FRAMES):
    f = base if i == 0 else render(i)
    f.save(os.path.join(frames_dir, f"f{i:03d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(frames_dir, "f%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT
], check=True)
print("horizontal lines:", horizontal)
print("wrote", OUT)
