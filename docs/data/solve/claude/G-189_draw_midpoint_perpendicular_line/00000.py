import numpy as np, subprocess, os
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N, FPS = 50, 16

base = Image.open(BASE).convert("RGB")
arr = np.array(base).astype(int)

# locate the two horizontal black lines
dark = (arr.sum(2) < 150)
rows = np.where(dark.sum(1) > arr.shape[1] * 0.8)[0]
gaps = np.where(np.diff(rows) > 1)[0]
upper = rows[: gaps[0] + 1]; lower = rows[gaps[0] + 1 :]
y_top, y_bot = int(upper.min()), int(lower.max())
y_mid = (upper.mean() + lower.mean()) / 2.0

# find the dot whose center lies at the vertical midpoint
colored = (np.abs(arr - arr.mean(2, keepdims=True)).sum(2) > 40)
from scipy import ndimage  # noqa
lab, n = ndimage.label(colored)
best = None
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(ys) < 50: continue
    d = abs(ys.mean() - y_mid)
    if best is None or d < best[0]:
        best = (d, xs.mean())
x_line = int(round(best[1]))

frames_dir = "/app/output/frames"
os.makedirs(frames_dir, exist_ok=True)
for f in os.listdir(frames_dir): os.remove(os.path.join(frames_dir, f))

for i in range(N):
    im = base.copy()
    t = i / (N - 1)
    if i > 0:
        # ease-in-out growth from top line to bottom line
        s = 0.5 - 0.5 * np.cos(np.pi * t)
        y_end = y_top + s * (y_bot - y_top)
        d = ImageDraw.Draw(im)
        d.line([(x_line, y_top), (x_line, y_end)], fill=(255, 0, 0), width=4)
    im.save(f"{frames_dir}/f_{i:03d}.png")

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", f"{frames_dir}/f_%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "12", "-r", str(FPS), OUT], check=True)
print("wrote", OUT, "line x =", x_line, "y", y_top, "->", y_bot)
