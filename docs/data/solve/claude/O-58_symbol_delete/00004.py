#!/usr/bin/env python3
"""Delete the orange hollow star at position 5: fade it out, then slide
symbols 6-9 one slot to the left. Everything else stays fixed."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 45
TARGET = 5                      # 1-based position to delete
N_CELLS = 9
PITCH = 105
BOX_X0 = 44                     # left border column of cell 1
BOX_W = 97                      # border-to-border width
BOX_Y0, BOX_Y1 = 464, 560       # top/bottom border rows
BG = np.array([255, 255, 255], dtype=np.uint8)

FADE_END = 20                   # frame index where the target is fully gone
SLIDE_START = 22                # sliding begins (short hold after fade)
SLIDE_END = N_FRAMES - 1        # final frame fully closed


def interior(i):
    """Interior pixel bounds (x0, x1, y0, y1 inclusive) of 1-based cell i."""
    bx0 = BOX_X0 + PITCH * (i - 1)
    return bx0 + 1, bx0 + BOX_W - 2, BOX_Y0 + 1, BOX_Y1 - 1


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = base.shape

    # Extract each symbol as (patch, mask) taken from its cell interior.
    symbols = {}
    for i in range(1, N_CELLS + 1):
        x0, x1, y0, y1 = interior(i)
        patch = base[y0:y1 + 1, x0:x1 + 1].copy()
        mask = (patch != BG).any(axis=2)
        symbols[i] = (patch, mask, x0, y0)

    # Static background: original frame with the target and all moving
    # symbols' interiors cleared to the background colour.
    static = base.copy()
    for i in range(TARGET, N_CELLS + 1):
        x0, x1, y0, y1 = interior(i)
        static[y0:y1 + 1, x0:x1 + 1] = BG

    def blit(img, patch, mask, x, y):
        h, w = mask.shape
        region = img[y:y + h, x:x + w]
        region[mask] = patch[mask]

    frames = []
    for f in range(N_FRAMES):
        img = static.copy()
        # Target symbol: fade to background colour.
        fade = smoothstep(f / FADE_END) if FADE_END > 0 else 1.0
        if fade < 1.0:
            patch, mask, x0, y0 = symbols[TARGET]
            faded = (patch.astype(np.float32) * (1 - fade)
                     + BG.astype(np.float32) * fade)
            blit(img, np.round(faded).astype(np.uint8), mask, x0, y0)
        # Remaining symbols to the right: slide left by one pitch.
        if f < SLIDE_START:
            s = 0.0
        else:
            s = smoothstep((f - SLIDE_START) / (SLIDE_END - SLIDE_START))
        dx = int(round(-PITCH * s))
        for i in range(TARGET + 1, N_CELLS + 1):
            patch, mask, x0, y0 = symbols[i]
            blit(img, patch, mask, x0 + dx, y0)
        frames.append(img)

    # Ensure first frame is exactly the source image.
    assert np.array_equal(frames[0], base), "first frame mismatch"

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0",
           "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames @ {FPS} fps")


if __name__ == "__main__":
    main()
