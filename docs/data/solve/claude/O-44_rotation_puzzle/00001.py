#!/usr/bin/env python3
"""Generate the pipe-puzzle solving video from first_frame.png.

Scene analysis (measured from first_frame.png):
  * Four 221x221 tiles with 3px borders; inner white areas span
    x/y 280..494 (left/top) and 530..744 (right/bottom).
  * Every pipe is the same L-shaped elbow (top-left tile = base, arms pointing
    right and down), rotated about the tile's inner centre:
        TL 0 deg, TR ~70 deg CW, BL 270 deg CW, BR ~119.2 deg CW.
  * Solved (closed loop): TL 0, TR 90, BR 180, BL 270 (deg CW).

All four pipes rotate clockwise simultaneously with an ease-in-out profile and
reach their solved orientations together on the last frame.  Nothing outside
the tiles' inner areas is touched.
"""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 96
SS = 4  # supersampling factor for smooth rotation

ORANGE = np.array([249, 115, 22], dtype=np.float32)
WHITE = np.array([255, 255, 255], dtype=np.float32)

# inner (white) area of each tile: (y0, x0), size 215x215
INNER = 215
TILES = {
    "TL": (280, 280),
    "TR": (280, 530),
    "BL": (530, 280),
    "BR": (530, 530),
}
# measured start angle (deg, clockwise) -> final angle (deg, clockwise)
ANGLES = {
    "TL": (0.0, 360.0),      # already correct: one full spin
    "TR": (70.0, 450.0),     # -> 90
    "BL": (270.0, 630.0),    # already correct: one full spin -> 270
    "BR": (119.2, 540.0),    # -> 180
}


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def main():
    first = np.array(Image.open(SRC).convert("RGB"))
    assert first.shape == (H, W, 3)

    # Base pipe sprite = alpha mask of the top-left tile (binary, exact).
    y0, x0 = TILES["TL"]
    tl = first[y0:y0 + INNER, x0:x0 + INNER].astype(np.float32)
    base = np.clip((255.0 - tl[..., 1]) / (255.0 - 115.0), 0, 1)
    base = (base > 0.5).astype(np.float32)
    big = cv2.resize(base, None, fx=SS, fy=SS, interpolation=cv2.INTER_NEAREST)
    c = (big.shape[1] / 2.0, big.shape[0] / 2.0)

    def rotated_alpha(theta_cw):
        theta_cw = theta_cw % 360.0
        q = round(theta_cw / 90.0)
        if abs(theta_cw - 90.0 * q) < 1e-6:
            # exact for multiples of 90 degrees (np.rot90 is counter-clockwise)
            return np.rot90(base, -q % 4).copy()
        M = cv2.getRotationMatrix2D(c, -theta_cw, 1.0)  # cv2 positive = CCW
        r = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        return cv2.resize(r, None, fx=1.0 / SS, fy=1.0 / SS, interpolation=cv2.INTER_AREA)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "12",
           "-pix_fmt", "yuv420p", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for i in range(N_FRAMES):
        if i == 0:
            frame = first.copy()
        else:
            t = smoothstep(i / (N_FRAMES - 1))
            frame = first.copy()
            for name, (ty, tx) in TILES.items():
                a0, a1 = ANGLES[name]
                alpha = rotated_alpha(a0 + (a1 - a0) * t)[..., None]
                region = WHITE * (1.0 - alpha) + ORANGE * alpha
                frame[ty:ty + INNER, tx:tx + INNER] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
