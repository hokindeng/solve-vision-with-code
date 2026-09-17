#!/usr/bin/env python3
"""Two balls move toward each other at equal speed and merge at the midpoint.
Overlapping regions use additive light mixing (clip(c1 + c2))."""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 80
RADIUS = 120
OUTLINE_W = 2
BG = (255, 255, 255)
OUT_DIR = "/app/output"
OUT_PATH = os.path.join(OUT_DIR, "video.mp4")

# Ball parameters measured from first_frame.png
BALLS = [
    {"center": (311.0, 569.0), "color": (108, 73, 107)},   # purple
    {"center": (708.0, 441.0), "color": (54, 112, 92)},    # green
]
MID = ((BALLS[0]["center"][0] + BALLS[1]["center"][0]) / 2,
       (BALLS[0]["center"][1] + BALLS[1]["center"][1]) / 2)


def ball_masks(cx, cy):
    """Return (fill_mask, outline_mask) for a ball at (cx, cy)."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS],
              fill=128, outline=255, width=OUTLINE_W)
    a = np.array(img)
    return a == 128, a == 255


def render(t):
    """t in [0,1]; 0 = start positions, 1 = fully merged at midpoint."""
    frame = np.full((H, W, 3), BG, dtype=np.float32)
    light = np.zeros((H, W, 3), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    outline = np.zeros((H, W), dtype=bool)
    for b in BALLS:
        cx = b["center"][0] + (MID[0] - b["center"][0]) * t
        cy = b["center"][1] + (MID[1] - b["center"][1]) * t
        fill, edge = ball_masks(cx, cy)
        light[fill] += np.array(b["color"], dtype=np.float32)
        covered |= fill
        outline |= edge
    frame[covered] = np.clip(light[covered], 0, 255)   # additive mixing
    frame[outline] = 0
    return frame.astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT_PATH]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        proc.stdin.write(render(t).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
