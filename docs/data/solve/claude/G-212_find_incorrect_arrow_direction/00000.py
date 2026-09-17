#!/usr/bin/env python3
"""Draw a red circle around the arrow pointing the wrong way (left vertical arrow).

The 10 arrows flow counterclockwise around the ring; the left-side arrow
points up instead of down.  We sweep a red circle around it over ~2.5 s,
then hold the finished result until 3.0 s.  All other pixels are untouched.
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 48
SWEEP_FRAMES = 40           # circle fully drawn by frame 40, then hold

# Odd arrow (measured from first_frame.png): x 141..200, y 429..593
CENTER = (170, 511)
RADIUS = 100
THICKNESS = 5
RED = (255, 0, 0)           # RGB
SS = 4                      # supersampling factor for anti-aliased stroke


def draw_arc(base: np.ndarray, frac: float) -> np.ndarray:
    """Return a copy of base with a red arc covering `frac` of the circle."""
    frame = base.copy()
    if frac <= 0:
        return frame
    pad = RADIUS + THICKNESS + 4
    cx, cy = CENTER
    x0, y0 = cx - pad, cy - pad
    x1, y1 = cx + pad, cy + pad
    patch = frame[y0:y1, x0:x1]
    h, w = patch.shape[:2]
    # Supersampled alpha mask for the arc
    mask = np.zeros((h * SS, w * SS), np.uint8)
    start = -90                           # start at top, go clockwise
    end = start + 360.0 * min(frac, 1.0)
    if frac >= 1.0:
        cv2.circle(mask, ((cx - x0) * SS, (cy - y0) * SS), RADIUS * SS, 255,
                   THICKNESS * SS, lineType=cv2.LINE_AA)
    else:
        cv2.ellipse(mask, ((cx - x0) * SS, (cy - y0) * SS),
                    (RADIUS * SS, RADIUS * SS), 0, start, end, 255,
                    THICKNESS * SS, lineType=cv2.LINE_AA)
    alpha = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    red = np.array(RED, np.float32)[None, None, :]
    blended = patch.astype(np.float32) * (1 - alpha) + red * alpha
    frame[y0:y1, x0:x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return frame


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    frames = []
    for i in range(N_FRAMES):
        t = min(i / (SWEEP_FRAMES - 1), 1.0)
        frames.append(draw_arc(base, ease(t)))

    # Encode with ffmpeg (H.264, yuv420p)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
