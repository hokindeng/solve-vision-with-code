#!/usr/bin/env python3
"""Generate the analogy video: crescent -> outline-only style -> move down.

Frame 0 is exactly first_frame.png.  Step 1 thins the crescent's 3px stroke to a
2px stroke (matching rhombus A -> B) by fading out the stroke's innermost pixel
layer.  Step 2 translates the thin crescent down by 60px (the same vertical
offset rhombus B -> C shows).  Every other pixel is left untouched.
"""
import os
import subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 64

# Crescent (bottom-left "D") geometry, measured from first_frame.png.
X0, X1, Y0, Y1 = 108, 270, 602, 762            # bounding box (inclusive)
OUTER_C, OUTER_R = (189.0, 682.0), 79.25       # ring centre / mid radius
INNER_C, INNER_R = (214.0, 682.0), 55.0
DY = 60                                        # rhombus B -> C vertical shift

# Timeline (frame indices)
T_HOLD0 = 6          # frames 0..5   : hold
T_STYLE_END = 30     # frames 6..29  : stroke thins (step 1)
T_MOVE_START = 34    # frames 30..33 : hold
T_MOVE_END = 60      # frames 34..59 : move down (step 2)
                     # frames 60..63 : hold final


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    assert base.shape == (H, W, 3)

    # --- isolate the crescent's green stroke pixels -------------------------
    green = (base[:, :, 0] < 128) & (base[:, :, 1] > 128) & (base[:, :, 2] < 128)
    region = np.zeros_like(green)
    region[Y0:Y1 + 1, X0:X1 + 1] = True
    ring = green & region
    stroke_color = base[ring][0].astype(np.float64)   # (38,191,38)
    bg = np.array([255, 255, 255], np.float64)

    # --- classify each stroke pixel by circle, find its inner-edge layer ----
    ys, xs = np.nonzero(ring)
    d_out = np.abs(np.hypot(xs - OUTER_C[0], ys - OUTER_C[1]) - OUTER_R)
    d_in = np.abs(np.hypot(xs - INNER_C[0], ys - INNER_C[1]) - INNER_R)
    inner_layer = np.zeros_like(ring)
    for x, y, do, di in zip(xs, ys, d_out, d_in):
        cx, cy = OUTER_C if do <= di else INNER_C
        vx, vy = cx - x, cy - y
        n = max(np.hypot(vx, vy), 1e-9)
        nx, ny = int(round(x + vx / n)), int(round(y + vy / n))
        if not ring[ny, nx]:          # neighbour toward centre is background
            inner_layer[y, x] = True
    thin_ring = ring & ~inner_layer

    # --- render frames -------------------------------------------------------
    frames = []
    for f in range(N_FRAMES):
        frame = base.copy()
        if f < T_HOLD0:
            frames.append(frame)
            continue

        # step 1: fade the inner stroke layer to background
        a = smoothstep((f - T_HOLD0 + 1) / (T_STYLE_END - T_HOLD0))
        if f >= T_STYLE_END:
            a = 1.0
        col = (1 - a) * stroke_color + a * bg
        frame[inner_layer] = np.round(col).astype(np.uint8)

        # step 2: translate the (thin) crescent down
        if f >= T_MOVE_START:
            m = smoothstep((f - T_MOVE_START + 1) / (T_MOVE_END - T_MOVE_START))
            dy = int(round(m * DY))
            frame[Y0:Y1 + 1, X0:X1 + 1] = 255          # clear original location
            shifted = np.zeros_like(thin_ring)
            shifted[Y0 + dy:Y1 + 1 + dy, X0:X1 + 1] = thin_ring[Y0:Y1 + 1, X0:X1 + 1]
            frame[shifted] = stroke_color.astype(np.uint8)

        frames.append(frame)

    # --- encode ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")

    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print(f"wrote {OUT}: {len(frames)} frames @ {FPS} fps")


if __name__ == "__main__":
    main()
