"""Predict the next color in the sequence: green, green, tan, tan -> green.
Fills the empty (gray-ring) fifth circle with green, growing radially over 4 s."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 64
GREEN = np.array([171, 244, 140], dtype=np.uint8)
CX, CY, R_IN = 912.0, 512.0, 71.0  # ring center and interior radius

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape
yy, xx = np.mgrid[0:H, 0:W]
dist = np.hypot(xx - CX, yy - CY)
interior = (dist <= R_IN) & np.all(base == 255, axis=2)  # white pixels inside ring only

def ease(t):  # smoothstep
    return t * t * (3 - 2 * t)

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-crf", "18", "-movflags", "+faststart", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(N):
    frame = base.copy()
    if i > 0:
        r = R_IN * ease(i / (N - 1)) + 0.5
        mask = interior & (dist <= r)
        frame[mask] = GREEN
    p.stdin.write(frame.tobytes())
p.stdin.close()
p.wait()
print("wrote", OUT)
