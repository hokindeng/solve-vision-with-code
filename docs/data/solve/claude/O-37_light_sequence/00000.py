#!/usr/bin/env python3
"""Generate the light-sequence video: turn lights so only the 6th is on.

Only the pixels of lights whose state changes are touched; everything else
is copied verbatim from first_frame.png in every frame.
"""
import os
import subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 35
TARGET_ON = {5}  # 6th light from the left (0-based index 5)

# Light geometry (measured from first_frame.png)
CX = [142 + 82 * k for k in range(10)]
CY = 511
R = 27  # patch half-size; covers glow (radius ~25) with margin


def light_state(im, cx):
    """True if the light centred at cx is on (yellow), False if gray."""
    return tuple(im[CY, cx]) == (255, 255, 0)


def patch(im, cx):
    return im[CY - R:CY + R + 1, cx - R:cx + R + 1].astype(np.float32)


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    states = [light_state(base, cx) for cx in CX]

    # Templates for the on and off appearance, taken from the frame itself.
    on_idx = [i for i, s in enumerate(states) if s]
    off_idx = [i for i, s in enumerate(states) if not s]
    if not on_idx or not off_idx:
        raise SystemExit("need at least one on and one off light for templates")
    tmpl_on = patch(base, CX[on_idx[0]])
    tmpl_off = patch(base, CX[off_idx[0]])

    changes = [i for i in range(10) if states[i] != (i in TARGET_ON)]

    # Schedule: transitions are staggered across the whole clip, each
    # lasting ~14 frames, with the last one finishing a couple of frames early.
    dur = 14
    last_start = N_FRAMES - 1 - dur - 2
    if len(changes) > 1:
        starts = [round(2 + (last_start - 2) * k / (len(changes) - 1)) for k in range(len(changes))]
    else:
        starts = [round((2 + last_start) / 2)]

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        im = base.copy()
        for i, s0 in zip(changes, starts):
            t = smoothstep((f - s0) / dur)
            src = tmpl_on if states[i] else tmpl_off
            dst = tmpl_off if states[i] else tmpl_on
            blend = src * (1 - t) + dst * t
            cx = CX[i]
            im[CY - R:CY + R + 1, cx - R:cx + R + 1] = np.clip(np.round(blend), 0, 255).astype(np.uint8)
        frames.append(im)

    # Sanity: first frame is untouched.
    assert (frames[0] == base).all()

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "medium",
        "-movflags", "+faststart", OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {N_FRAMES} frames @ {FPS} fps; changed lights {[i + 1 for i in changes]}")


if __name__ == "__main__":
    main()
