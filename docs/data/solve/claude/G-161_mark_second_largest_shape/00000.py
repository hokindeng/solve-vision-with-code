import numpy as np, subprocess, os, math
from PIL import Image, ImageDraw
from scipy import ndimage

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FRAMES, FPS = 40, 16

base = Image.open(SRC).convert("RGB")
arr = np.array(base)

# Detect shapes: connected components of non-background (non-white) pixels
mask = np.any(arr < 240, axis=2)
mask = ndimage.binary_closing(mask, iterations=2)
labels, n = ndimage.label(mask)
shapes = []
for i in range(1, n + 1):
    ys, xs = np.nonzero(labels == i)
    area = len(xs)
    if area < 500:
        continue
    cx, cy = xs.mean(), ys.mean()
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
    shapes.append((area, cx, cy, r))
shapes.sort(reverse=True)
assert len(shapes) == 3, shapes
area, cx, cy, r = shapes[1]  # second largest
print("shapes (area,cx,cy,r):", shapes)
print("second largest:", (cx, cy, r))

ring_r = r + 22
width = 8
RED = (255, 0, 0)

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)

os.makedirs("/app/output", exist_ok=True)
tmp = "/app/output/frames"
os.makedirs(tmp, exist_ok=True)
for f in range(FRAMES):
    im = base.copy()
    if f > 0:
        # sweep from frame 1 to frame 33, then hold complete ring
        t = min(1.0, (f - 1) / 32.0)
        sweep = 360.0 * ease(t)
        if sweep > 0:
            # supersample for a smooth ring
            S = 4
            big = Image.new("RGBA", (im.width * S, im.height * S), (0, 0, 0, 0))
            d = ImageDraw.Draw(big)
            bbox = [(cx - ring_r) * S, (cy - ring_r) * S, (cx + ring_r) * S, (cy + ring_r) * S]
            start = -90
            d.arc(bbox, start, start + sweep, fill=RED + (255,), width=width * S)
            # round caps
            for ang in (start, start + sweep):
                a = math.radians(ang)
                rc = ring_r - width / 2.0  # PIL arc width extends inward from bbox
                px, py = (cx + rc * math.cos(a)) * S, (cy + rc * math.sin(a)) * S
                rr = width * S / 2
                d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=RED + (255,))
            ov = big.resize(im.size, Image.LANCZOS)
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    im.save(f"{tmp}/{f:03d}.png")

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", f"{tmp}/%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "12", "-r", str(FPS), OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", OUT)
