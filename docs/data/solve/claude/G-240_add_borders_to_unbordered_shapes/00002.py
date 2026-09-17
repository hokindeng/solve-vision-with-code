"""Add a thin black border to every unbordered shape, animated over 80 frames."""
import subprocess, numpy as np, cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS, NFRAMES = 16, 80
THICK = 5.0  # matches the existing border (inside the shape edge)

base = np.array(Image.open(SRC).convert("RGB"))
bg = base[0, 0].copy()
nonbg = (base != bg).any(2).astype(np.uint8)
n, lab, stats, cents = cv2.connectedComponentsWithStats(nonbg, 8)

# Border pixels to add: for each shape lacking a black outline, the pixels
# inside the shape within THICK of the shape's outer edge.
targets = []  # (mask of new border pixels, angle map around centroid)
for i in range(1, n):
    m = (lab == i)
    black = m & (base == 0).all(2)
    if black.sum() > 0.02 * m.sum():
        continue  # already bordered
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    border = m & (dt < THICK)
    cy, cx = cents[i][1], cents[i][0]
    ys, xs = np.nonzero(border)
    ang = (np.arctan2(ys - cy, xs - cx) + np.pi / 2) % (2 * np.pi)  # start at top, clockwise
    targets.append((ys, xs, ang))

def frame(t):
    """t in [0,1]: fraction of the border traced around each shape."""
    img = base.copy()
    for ys, xs, ang in targets:
        sel = ang <= t * 2 * np.pi + 1e-9
        img[ys[sel], xs[sel]] = 0
    return img

proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for k in range(NFRAMES):
    # frame 0 is untouched; the border is traced over frames 1..NFRAMES-6 and holds.
    t = 0.0 if k == 0 else min(1.0, k / (NFRAMES - 6))
    proc.stdin.write(frame(t).tobytes())
proc.stdin.close(); proc.wait()
Image.fromarray(frame(1.0)).save("/app/output/last_frame.png")
print("wrote", OUT)
