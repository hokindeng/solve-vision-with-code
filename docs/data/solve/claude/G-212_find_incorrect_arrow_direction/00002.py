import numpy as np
from PIL import Image, ImageDraw
import subprocess, os
from scipy import ndimage

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 48

base = Image.open(SRC).convert("RGB")
W, H = base.size
arr = np.array(base).astype(int)

# --- find arrows: non-background pixels, connected components ---
bg = arr[0, 0]
mask = np.abs(arr - bg).sum(axis=2) > 60
mask = ndimage.binary_dilation(mask, iterations=3)
lab, n = ndimage.label(mask)
comps = []
for i in range(1, n + 1):
    ys, xs = np.nonzero(lab == i)
    if len(xs) < 200:
        continue
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    # principal axis
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]
    proj = (pts - c) @ axis
    # arrowhead end has more pixels (extra strokes): compare pixel counts at both ends
    lo, hi = proj.min(), proj.max()
    end_a = (proj < lo + 0.3 * (hi - lo)).sum()
    end_b = (proj > hi - 0.3 * (hi - lo)).sum()
    d = axis if end_b > end_a else -axis
    comps.append(dict(center=c, dir=d, bbox=(xs.min(), ys.min(), xs.max(), ys.max())))

cx, cy = np.mean([c["center"] for c in comps], axis=0)
signs = []
for c in comps:
    r = c["center"] - np.array([cx, cy])
    signs.append(np.sign(r[0] * c["dir"][1] - r[1] * c["dir"][0]))
signs = np.array(signs)
majority = 1 if (signs > 0).sum() > (signs < 0).sum() else -1
odd = [c for c, s in zip(comps, signs) if s != majority]
assert len(odd) == 1, f"expected one odd arrow, got {len(odd)}"
odd = odd[0]
x0, y0, x1, y1 = odd["bbox"]
ocx, ocy = (x0 + x1) / 2, (y0 + y1) / 2
radius = int(np.hypot(x1 - x0, y1 - y0) / 2 + 22)
print("odd arrow center", (ocx, ocy), "radius", radius)

# --- render frames: circle sweeps in over the duration (supersampled AA) ---
SS = 4
LW = 6
frames = []
for f in range(N):
    t = f / (N - 1)
    frame = base.copy()
    if t > 0:
        ov = Image.new("L", (W * SS, H * SS), 0)
        d = ImageDraw.Draw(ov)
        box = [(ocx - radius) * SS, (ocy - radius) * SS, (ocx + radius) * SS, (ocy + radius) * SS]
        sweep = min(360.0, 360.0 * t)
        start = -90
        if sweep >= 360:
            d.ellipse(box, outline=255, width=LW * SS)
        else:
            d.arc(box, start, start + sweep, fill=255, width=LW * SS)
        a = ov.resize((W, H), Image.LANCZOS)
        red = Image.new("RGB", (W, H), (220, 30, 30))
        frame = Image.composite(red, frame, a)
    frames.append(frame)

# first frame must match exactly
assert np.array_equal(np.array(frames[0]), np.array(base))

tmp = "/app/output/frames"
os.makedirs(tmp, exist_ok=True)
for i, fr in enumerate(frames):
    fr.save(f"{tmp}/{i:03d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%03d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", OUT)
