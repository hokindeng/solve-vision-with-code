#!/usr/bin/env python3
"""Substitute the hollow square at position 4 with an orange solid circle.

Animation: the old symbol fades out into white, then the new symbol fades in
from white at the same position. Everything else stays identical to first_frame.png.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 52
ORANGE = np.array([255, 165, 0], dtype=np.float32)
WHITE = np.array([255, 255, 255], dtype=np.float32)

# Cell 4 outer border: x 411..507, y 464..560  -> center (459, 512)
CELL4 = (slice(465, 560), slice(412, 507))  # interior (excluding grey border)
# Cell 7 (existing solid circle, same size as the reference symbol): center (774, 512)
CELL7 = (slice(465, 560), slice(727, 822))
CX4, CY4 = 459, 512
CIRCLE_R = 34.25  # fallback if the template mask cannot be recovered


def ease(t):
    """Smooth ease-in-out on [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
    H, W, _ = base.shape

    # --- old symbol: the hollow orange square inside cell 4 -----------------
    cell = base[CELL4]
    old_mask = ~np.all(cell == WHITE, axis=2)  # non-white pixels inside the cell
    old_layer = cell.copy()

    # --- new symbol: solid orange circle, same geometry as existing circles ---
    cell7 = base[CELL7]
    tmpl = np.all(cell7 == np.array([128, 0, 0], dtype=np.float32), axis=2)
    if tmpl.sum() > 3000:  # template recovered from the maroon circle at position 7
        new_mask = tmpl
    else:  # fallback: analytic circle centred in the cell
        ys, xs = np.mgrid[CELL4[0], CELL4[1]]
        new_mask = (xs - CX4) ** 2 + (ys - CY4) ** 2 <= CIRCLE_R ** 2
    new_layer = np.where(new_mask[..., None], ORANGE, WHITE)

    half = N_FRAMES // 2  # frames 0..25 fade out, 26..51 fade in
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        region = f[CELL4]
        if i < half:
            a = 1.0 - ease(i / (half - 1))  # 1 at frame 0 (exact first frame) -> 0
            blended = WHITE + (old_layer - WHITE) * a
            region[old_mask] = blended[old_mask]
        else:
            a = ease((i - half + 1) / (N_FRAMES - half))  # ->1 at last frame
            blended = WHITE + (new_layer - WHITE) * a
            # old symbol fully gone: cell interior is white except new circle
            region[old_mask] = WHITE
            region[new_mask] = blended[new_mask]
        f[CELL4] = region
        frames.append(np.clip(np.round(f), 0, 255).astype(np.uint8))

    # sanity: first frame identical to input, last frame shows the circle
    assert np.array_equal(frames[0], base.astype(np.uint8))

    # --- encode with ffmpeg (H.264, yuv420p, 16 fps) -----------------------
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {N_FRAMES} frames @ {FPS} fps")


if __name__ == "__main__":
    main()
