#!/usr/bin/env python3
"""Draw a red circle around the arrow pointing in a different direction.

The eight arrows in first_frame.png follow a clockwise tangential pattern
except the bottom-right one (bbox x 534..707, y 767..858), which points
up-right instead of down-left.  The circle is stroked progressively over
the 3-second clip; every other pixel remains identical to the first frame.
"""
import os
import subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 48
W = H = 1024

# Odd arrow: centre of its bounding box, radius large enough to enclose it.
CX, CY = 620.5, 812.5
RADIUS = 118
THICKNESS = 6
RED = (0, 0, 255)  # BGR
START_ANGLE = -90.0  # start stroking from the top of the circle
DRAW_FRAMES = 42     # frames spent drawing; the rest hold the finished result
SS = 4               # supersampling factor for a smooth anti-aliased stroke


def ease(t: float) -> float:
    return 0.5 - 0.5 * np.cos(np.pi * t)


def render_frame(base: np.ndarray, sweep_deg: float) -> np.ndarray:
    if sweep_deg <= 0:
        return base.copy()
    # Render the arc on a supersampled alpha mask, then composite red over base.
    mask = np.zeros((H * SS, W * SS), np.uint8)
    cv2.ellipse(
        mask,
        (int(round(CX * SS)), int(round(CY * SS))),
        (RADIUS * SS, RADIUS * SS),
        0.0,
        START_ANGLE,
        START_ANGLE + min(sweep_deg, 360.0),
        255,
        THICKNESS * SS,
        cv2.LINE_AA,
    )
    alpha = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    red = np.empty_like(base, dtype=np.float32)
    red[:] = RED
    out = base.astype(np.float32) * (1 - alpha) + red * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def bgr_to_yuv420p(frame: np.ndarray) -> bytes:
    """Exact BT.601 limited-range conversion (ffmpeg's default swscale path
    rounds white to ~(250,253,251); doing it ourselves keeps it at 255)."""
    f = frame.astype(np.float64)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255
    u = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255
    v = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255

    def q(a):
        return np.clip(np.round(a), 0, 255).astype(np.uint8)

    def sub(c):
        return q(c.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3)))

    return q(y).tobytes() + sub(u).tobytes() + sub(v).tobytes()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    base = cv2.imread(SRC, cv2.IMREAD_COLOR)
    assert base is not None and base.shape[:2] == (H, W), "unexpected first frame"

    frames = []
    for i in range(N_FRAMES):
        t = min(i / DRAW_FRAMES, 1.0)
        sweep = 360.0 * ease(t)
        frames.append(render_frame(base, sweep))

    # Frame 0 must be exactly the first frame.
    assert np.array_equal(frames[0], base)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryslow", "-crf", "6", "-tune", "stillimage",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(bgr_to_yuv420p(f))
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({N_FRAMES} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
