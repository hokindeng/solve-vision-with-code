#!/usr/bin/env python3
"""Insert an orange hollow diamond at slot 5 (1-indexed).

Phase 1: the symbol in slot 5 (olive square) slides right into slot 6.
Phase 2: the new diamond fades in above slot 5 and slides down into it.
Everything else stays identical to first_frame.png.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 32
FPS = 16

# Slot geometry measured from first_frame.png (boxes are 97 px wide, pitch 105).
SLOT_X0 = 202          # left border column of slot 1
PITCH = 105
BOX_Y0, BOX_Y1 = 464, 560   # border rows
CY = (BOX_Y0 + BOX_Y1) / 2.0  # 512
TARGET = 4             # 0-indexed slot 5
ORANGE = (255, 165, 0)
DIAMOND_HALF = 39      # same size as the reference-panel diamond
LINE_W = 3


def slot_center_x(i):
    return SLOT_X0 + i * PITCH + 48.0


def interior(i):
    """x0, x1 (exclusive), y0, y1 (exclusive) of the slot's white interior."""
    x0 = SLOT_X0 + i * PITCH + 1
    return x0, x0 + 95, BOX_Y0 + 1, BOX_Y1


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def render_diamond(cx, cy, alpha, size=1024):
    """Return a float mask (H,W) of the hollow diamond.

    Drawn the same way as the reference-panel symbol: a PIL line loop of width
    LINE_W at native resolution (crisp, non-anti-aliased, 3 px cross-section).
    """
    cx, cy = int(round(cx)), int(round(cy))
    h = int(round(DIAMOND_HALF))
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    pts = [(cx, cy - h), (cx + h, cy), (cx, cy + h), (cx - h, cy), (cx, cy - h)]
    d.line(pts, fill=255, width=LINE_W)
    return np.asarray(img, dtype=np.float32) / 255.0 * alpha


def composite(base, mask, color):
    out = base.astype(np.float32)
    m = mask[..., None]
    col = np.array(color, dtype=np.float32)[None, None, :]
    return out * (1 - m) + col * m


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # Extract the symbol currently in the target slot as a patch with alpha.
    x0, x1, y0, y1 = interior(TARGET)
    patch = base[y0:y1, x0:x1].astype(np.float32)
    # alpha: how far from white (the interior background) each pixel is
    patch_alpha = np.clip((255 - patch.min(axis=2)) / 255.0, 0, 1)
    patch_alpha = (patch_alpha > 0.02).astype(np.float32) * np.clip(patch_alpha * 4, 0, 1)
    # symbol colour with background removed (unpremultiply against white)
    a = np.clip(patch_alpha, 1e-3, 1)[..., None]
    patch_rgb = np.clip((patch - 255.0 * (1 - a)) / a, 0, 255)

    # Background with the target slot interior cleared (slot 6 interior is already empty).
    cleared = base.copy()
    cleared[y0:y1, x0:x1] = 255

    frames = []
    slide_end = 14          # frames 0..slide_end : olive square slides right
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(base.copy())
            continue
        img = cleared.astype(np.float32)

        # Phase 1: slide existing symbol from slot TARGET to TARGET+1
        t1 = ease(f / slide_end)
        dx = int(round(t1 * PITCH))
        m = np.zeros((H, W), np.float32)
        rgb = np.zeros((H, W, 3), np.float32)
        m[y0:y1, x0 + dx:x1 + dx] = patch_alpha
        rgb[y0:y1, x0 + dx:x1 + dx] = patch_rgb
        img = img * (1 - m[..., None]) + rgb * m[..., None]

        # Phase 2: new diamond fades in above the gap and drops into the slot
        if f > slide_end:
            t2 = (f - slide_end) / (N_FRAMES - 1 - slide_end)
            alpha = ease(min(t2 / 0.5, 1.0))
            drop = ease(t2)
            cy = (CY - 90.0) + drop * 90.0
            dm = render_diamond(slot_center_x(TARGET), cy, alpha)
            img = composite(img, dm, ORANGE)

        frames.append(np.clip(np.round(img), 0, 255).astype(np.uint8))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(tmp, "%04d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "8",
           "-pix_fmt", "yuv420p", "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    for i in range(len(frames)):
        os.remove(os.path.join(tmp, f"{i:04d}.png"))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
