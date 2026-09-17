"""Animate the clock from 6:00 to 9:00 (15 hours later): hour hand sweeps
450 deg clockwise, minute hand completes 15 turns. Everything else is kept
pixel-identical to first_frame.png."""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES, FPS, SS = 120, 16, 4          # SS = supersampling factor
CX, CY = 512.0, 512.0
BG = (255, 250, 205)
MIN_COL, MIN_LEN, MIN_W = (147, 112, 219), 287, 5
HR_COL, HR_LEN, HR_W = (139, 69, 19), 204, 8
HUB_COL, HUB_R = (0, 0, 0), 10.5
CLEAR_R = 300                              # hands live well inside the tick ring

HOURS = 15.0
HR_START, MIN_START = 180.0, 0.0           # 6:00 -> hour hand down, minute hand up
HR_SWEEP, MIN_SWEEP = HOURS * 30.0, HOURS * 360.0


def ease(t):  # smooth start/stop
    return t * t * (3 - 2 * t)


def hand_polygon(angle_deg, length, width, back=0.0):
    a = math.radians(angle_deg)
    dx, dy = math.sin(a), -math.cos(a)          # 0 deg = up, clockwise positive
    nx, ny = -dy, dx
    hw = width / 2.0
    pts = [(-back, -hw), (length, -hw), (length, hw), (-back, hw)]
    return [((CX + u * dx + v * nx) * SS, (CY + u * dy + v * ny) * SS) for u, v in pts]


def render(base, hr_angle, min_angle):
    big = Image.new("RGB", (1024 * SS, 1024 * SS), BG)
    d = ImageDraw.Draw(big)
    d.polygon(hand_polygon(hr_angle, HR_LEN, HR_W), fill=HR_COL)
    d.polygon(hand_polygon(min_angle, MIN_LEN, MIN_W), fill=MIN_COL)
    r = HUB_R * SS
    d.ellipse([CX * SS - r, CY * SS - r, CX * SS + r, CY * SS + r], fill=HUB_COL)
    layer = np.asarray(big.resize((1024, 1024), Image.LANCZOS)).astype(np.float32)
    # composite only inside the cleared disc
    frame = base.copy().astype(np.float32)
    frame[mask] = layer[mask]
    return np.clip(frame + 0.5, 0, 255).astype(np.uint8)


first = np.asarray(Image.open(SRC).convert("RGB"))
yy, xx = np.mgrid[0:1024, 0:1024]
mask = (xx - CX) ** 2 + (yy - CY) ** 2 <= CLEAR_R ** 2
base = first.copy()
base[mask] = BG

os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", "1024x1024", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-crf", "16", "-preset", "medium", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(N_FRAMES):
    if i == 0:
        frame = first                                  # exact first frame
    else:
        t = ease(i / (N_FRAMES - 1))
        frame = render(base, HR_START + HR_SWEEP * t, MIN_START + MIN_SWEEP * t)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
proc.wait()
print("wrote", OUT)
