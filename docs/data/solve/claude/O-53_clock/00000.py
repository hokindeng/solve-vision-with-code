"""Animate the clock in first_frame.png forward by 17 hours (7:44 -> 12:44)."""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 120
CENTER = (512, 512)
BG = (245, 245, 220)
HOUR_COL, MIN_COL = (139, 69, 19), (147, 112, 219)
HOUR_LEN, MIN_LEN = 205, 285
HOUR_W, MIN_W = 8, 5
DOT_R = 10

START_H, START_M = 7, 44
ELAPSED_HOURS = 17


def ease(p):  # smooth ease-in-out
    return 0.5 - 0.5 * math.cos(math.pi * p)


def draw_hands(base, hours_elapsed):
    im = base.copy()
    d = ImageDraw.Draw(im)
    cx, cy = CENTER
    t_min = START_H * 60 + START_M + hours_elapsed * 60.0
    hour_ang = (t_min / 60.0) * 30.0
    min_ang = (t_min % 60.0) * 6.0
    for ang, L, w, col in [(hour_ang, HOUR_LEN, HOUR_W, HOUR_COL),
                           (min_ang, MIN_LEN, MIN_W, MIN_COL)]:
        r = math.radians(ang)
        d.line([(cx, cy), (cx + L * math.sin(r), cy - L * math.cos(r))], fill=col, width=w)
    d.ellipse([cx - DOT_R, cy - DOT_R, cx + DOT_R, cy + DOT_R], fill=(0, 0, 0))
    return im


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = Image.open(SRC).convert("RGB")
    arr = np.array(first)
    # Erase the existing hands (exact-colour match, image has no anti-aliasing).
    hand_mask = np.all(arr == HOUR_COL, axis=2) | np.all(arr == MIN_COL, axis=2)
    clean = arr.copy()
    clean[hand_mask] = BG
    base = Image.fromarray(clean)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        if i == 0:
            frame = first
        else:
            p = ease(i / (N_FRAMES - 1))
            frame = draw_hands(base, ELAPSED_HOURS * p)
        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
