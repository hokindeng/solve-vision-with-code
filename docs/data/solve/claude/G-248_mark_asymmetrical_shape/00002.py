import numpy as np, cv2, subprocess, os
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 16

img = np.array(Image.open(BASE).convert("RGB"))
H, W = img.shape[:2]

# --- find shapes (non-white components) ---
mask = (np.abs(img.astype(int) - 255).sum(axis=2) > 30).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

def asym_score(m):
    """Min over reflection axes through centroid of (1 - IoU(mask, mirrored mask))."""
    m = m.astype(np.uint8)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    best = 1.0
    for ang in np.arange(0, 180, 1.0):
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rot = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
        # reflect across vertical line x = cx
        M2 = np.float32([[-1, 0, 2 * cx], [0, 1, 0]])
        ref = cv2.warpAffine(rot, M2, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
        inter = np.logical_and(rot, ref).sum(); union = np.logical_or(rot, ref).sum()
        best = min(best, 1 - inter / union)
    return best

cands = []
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < 200:
        continue
    m = (lab == i)
    x, y, w, h = stats[i, :4]
    cands.append((asym_score(m), i, x, y, w, h))
cands.sort(reverse=True)
_, i, x, y, w, h = cands[0]
cx, cy = x + w / 2, y + h / 2
r = int(np.hypot(w, h) / 2 + 18)
print("asymmetric shape bbox", (x, y, w, h), "scores", [round(c[0], 3) for c in cands])

# --- render frames: red circle sweeps progressively ---
os.makedirs("/app/output", exist_ok=True)
frames = []
for k in range(N):
    fr = Image.fromarray(img.copy())
    if k > 0:
        t = k / (N - 1)
        d = ImageDraw.Draw(fr)
        end = -90 + 360 * t
        if t >= 1:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=5)
        else:
            d.arc([cx - r, cy - r, cx + r, cy + r], start=-90, end=end, fill=(255, 0, 0), width=5)
    frames.append(np.array(fr))

raw = "/app/output/frames.raw"
np.concatenate(frames).tofile(raw)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", OUT], check=True)
os.remove(raw)
print("wrote", OUT)
