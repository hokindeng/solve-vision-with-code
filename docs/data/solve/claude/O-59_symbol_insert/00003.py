#!/usr/bin/env python3
"""Insert a green solid down triangle at position 1.

Phase 1: symbols in slots 1..8 slide right one slot (into 2..9).
Phase 2: the green triangle (copied from the reference panel) fades in above
slot 1 and slides down into it.  Everything else is left untouched.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 24
SLIDE_FRAMES = 12          # frames 0..11: slide right
                           # frames 11..23: fade in + drop

# Slot geometry (measured from first_frame.png)
SLOT_X0 = 44               # left border column of slot 1
SLOT_Y0 = 464              # top border row
SLOT_W = 97                # incl. borders
PITCH = 105
N_SLOTS = 9
TARGET = 0                 # position 1 (0-based)

# Reference panel triangle bbox (solid, no anti-aliasing)
REF_X0, REF_X1, REF_Y0, REF_Y1 = 912, 979, 57, 115


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def interior(i):
    """(x0, x1, y0, y1) inclusive-exclusive interior of slot i."""
    x0 = SLOT_X0 + PITCH * i + 1
    return x0, x0 + SLOT_W - 2, SLOT_Y0 + 1, SLOT_Y0 + SLOT_W - 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    orig = np.array(Image.open(SRC).convert("RGB"))

    # Extract symbol patches and a background with all slot interiors cleared.
    base = orig.copy()
    patches = []
    for i in range(N_SLOTS):
        x0, x1, y0, y1 = interior(i)
        patch = orig[y0:y1, x0:x1].copy()
        mask = (patch.astype(int).sum(2) < 765 - 3)
        patches.append((patch, mask) if mask.any() else None)
        base[y0:y1, x0:x1] = 255

    # Green triangle patch from the reference panel.
    tri = orig[REF_Y0:REF_Y1 + 1, REF_X0:REF_X1 + 1].copy()
    tri_mask = (tri.astype(int).sum(2) < 765 - 3)
    tri_h, tri_w = tri_mask.shape
    # Destination: horizontally centred in slot 1, circumcentre at slot centre.
    slot_cx = SLOT_X0 + PITCH * TARGET + SLOT_W / 2.0          # 92.5
    slot_cy = SLOT_Y0 + SLOT_W / 2.0                           # 512.5
    R = (REF_Y1 - REF_Y0 + 1) / 1.5
    dst_x = int(round(slot_cx - tri_w / 2.0))
    dst_y_final = int(round(slot_cy - R / 2.0))
    dst_y_start = dst_y_final - 90

    def paste(img, patch, mask, x, y, alpha=1.0):
        h, w = mask.shape
        region = img[y:y + h, x:x + w]
        if alpha >= 1.0:
            region[mask] = patch[mask]
        else:
            blend = (region.astype(float) * (1 - alpha) + patch.astype(float) * alpha)
            region[mask] = np.clip(blend[mask] + 0.5, 0, 255).astype(np.uint8)

    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        # Phase 1: slide
        t1 = smoothstep(f / (SLIDE_FRAMES - 1))
        dx = int(round(t1 * PITCH))
        for i in range(N_SLOTS):
            if patches[i] is None:
                continue
            patch, mask = patches[i]
            x0, _, y0, _ = interior(i)
            shift = dx if i >= TARGET else 0
            paste(img, patch, mask, x0 + shift, y0)
        # Phase 2: fade in + drop
        if f >= SLIDE_FRAMES - 1:
            t2 = (f - (SLIDE_FRAMES - 1)) / (N_FRAMES - SLIDE_FRAMES)
            if f == N_FRAMES - 1:
                t2 = 1.0
            alpha = min(1.0, t2 / 0.55)            # fully opaque a bit before landing
            y = int(round(dst_y_start + smoothstep(t2) * (dst_y_final - dst_y_start)))
            if alpha > 0:
                paste(img, tri, tri_mask, dst_x, y, alpha)
        frames.append(img)

    assert np.array_equal(frames[0], orig), "first frame must match first_frame.png"

    # Encode with ffmpeg: H.264, yuv420p, 16 fps.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
