"""Generate the analogy video: recolor the plus red->green, then move it down 100 px.

The top row demonstrates: red minus -> green minus -> green minus shifted down by 100 px.
The bottom row's plus gets the same two-step change, applied sequentially. Everything else
(background, top row, arrows, question marks) is left byte-identical to first_frame.png.
"""
import os
import subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 60
FPS = 16

RED = np.array([255, 0, 0], dtype=np.float64)
GREEN = np.array([70, 153, 53], dtype=np.float64)  # colour demonstrated on the top row
DY = 100  # top-row minus moved from y=321 to y=421 -> same shift for the plus

# Timeline (frame indices, inclusive ranges)
HOLD0_END = 4        # frames 0..4  : unchanged start
RECOLOR_END = 28     # frames 5..28 : recolor red -> green
HOLD1_END = 32       # frames 29..32: hold recolored
MOVE_END = 56        # frames 33..56: move down by DY
                     # frames 57..59: hold final


def ease(t):
    """Smoothstep ease-in-out on [0,1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # Locate the plus: the red/black object in the lower half of the image.
    lower = np.zeros((H, W), dtype=bool)
    lower[500:] = True
    nonwhite = (base != 255).any(axis=2) & lower
    ys, xs = np.nonzero(nonwhite)
    # Restrict to the leftmost object (the plus); the '?' marks are further right.
    x0 = xs.min()
    sel = xs < x0 + 200
    ys, xs = ys[sel], xs[sel]
    y0, y1, x1 = ys.min(), ys.max(), xs.max()

    sprite = base[y0:y1 + 1, x0:x1 + 1].copy()
    sprite_mask = (sprite != 255).any(axis=2)               # object pixels (fill + outline)
    fill_mask = np.all(sprite == RED.astype(np.uint8), axis=2)  # red fill only

    # Background with the plus erased.
    bg = base.copy()
    bg[y0:y1 + 1, x0:x1 + 1][sprite_mask] = 255

    frames = []
    for i in range(N_FRAMES):
        if i <= HOLD0_END:
            c, d = 0.0, 0
        elif i <= RECOLOR_END:
            c = ease((i - HOLD0_END) / (RECOLOR_END - HOLD0_END))
            d = 0
        elif i <= HOLD1_END:
            c, d = 1.0, 0
        elif i <= MOVE_END:
            c = 1.0
            d = int(round(DY * ease((i - HOLD1_END) / (MOVE_END - HOLD1_END))))
        else:
            c, d = 1.0, DY

        spr = sprite.astype(np.float64).copy()
        spr[fill_mask] = (1 - c) * RED + c * GREEN
        spr = np.clip(np.rint(spr), 0, 255).astype(np.uint8)

        frame = bg.copy()
        region = frame[y0 + d:y1 + 1 + d, x0:x1 + 1]
        region[sprite_mask] = spr[sprite_mask]
        frames.append(frame)

    # Frame 0 must equal the input exactly.
    assert np.array_equal(frames[0], base)

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-x264-params", "keyint=16",
        OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames @ {FPS} fps, {W}x{H}")


if __name__ == "__main__":
    main()
