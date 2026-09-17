#!/usr/bin/env python3
"""Generate /app/output/video.mp4: insert two solid hearts at positions 2 and 6.

Sequence in first_frame.png: [heart, circle, star, triangle, ring, _, _, _, _]
Insert heart at position 2  -> [heart, HEART, circle, star, triangle, ring, _, _, _]
Insert heart at position 6  -> [heart, HEART, circle, star, triangle, HEART, ring, _, _]
Animation: existing symbols slide right one slot, then the heart flies in from the
reference panel (top-right) and settles into the freed cell.  All pixels outside the
animated elements are taken verbatim from first_frame.png.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 54

# Cell geometry measured from first_frame.png
N_CELLS = 9
CELL_X0 = 44          # left border column of cell 1
PITCH = 105           # distance between successive cell left borders
CELL_W = 97           # including both 1px borders (44..140)
CELL_Y0 = 464         # top border row
CELL_H = 97           # 464..560
INNER = 95            # interior size (45..139, 465..559)

# Reference panel heart (top-right) bounding box interior
REF_BOX = (887, 18, 1006, 137)  # x0,y0,x1,y1 inclusive


def cell_interior(i):
    """Return (x0, y0, x1, y1) exclusive-end slice of cell i (0-based) interior."""
    x0 = CELL_X0 + PITCH * i + 1
    y0 = CELL_Y0 + 1
    return x0, y0, x0 + INNER, y0 + INNER


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))

    # Extract sprites (RGBA) from the 5 filled cells.
    sprites = {}
    for i in range(5):
        x0, y0, x1, y1 = cell_interior(i)
        patch = base[y0:y1, x0:x1]
        mask = (patch != 255).any(axis=2)
        rgba = np.dstack([patch, (mask * 255).astype(np.uint8)])
        sprites[i] = rgba
    heart = sprites[0]

    # Background: first frame with all cell interiors blanked to white.
    background = base.copy()
    for i in range(N_CELLS):
        x0, y0, x1, y1 = cell_interior(i)
        background[y0:y1, x0:x1] = 255

    # Border pixel mask (gray cell frames) to re-stamp after sliding sprites.
    border_mask = np.zeros((H, W), dtype=bool)
    for i in range(N_CELLS):
        xl = CELL_X0 + PITCH * i
        xr = xl + CELL_W - 1
        border_mask[CELL_Y0:CELL_Y0 + CELL_H, xl] = True
        border_mask[CELL_Y0:CELL_Y0 + CELL_H, xr] = True
        border_mask[CELL_Y0, xl:xr + 1] = True
        border_mask[CELL_Y0 + CELL_H - 1, xl:xr + 1] = True

    def paste(canvas, rgba, x, y, alpha=1.0):
        """Alpha-composite an RGBA sprite at integer top-left (x, y)."""
        h, w = rgba.shape[:2]
        xs, ys = max(0, x), max(0, y)
        xe, ye = min(W, x + w), min(H, y + h)
        if xs >= xe or ys >= ye:
            return
        sub = rgba[ys - y:ye - y, xs - x:xe - x]
        a = (sub[..., 3:4].astype(np.float32) / 255.0) * alpha
        region = canvas[ys:ye, xs:xe].astype(np.float32)
        canvas[ys:ye, xs:xe] = (region * (1 - a) + sub[..., :3] * a).round().astype(np.uint8)

    # Sequence state: list of sprite indices per slot (None = empty)
    def slot_xy(slot_float):
        x0, y0, _, _ = cell_interior(0)
        return int(round(x0 + PITCH * slot_float)), y0

    ref_x = REF_BOX[0] + (REF_BOX[2] - REF_BOX[0] + 1 - INNER) // 2 + 1
    ref_y = REF_BOX[1] + (REF_BOX[3] - REF_BOX[1] + 1 - INNER) // 2 + 1

    # Timeline (frame indices)
    # Phase A: insert at position 2 (0-based slot 1)
    A_SHIFT = (4, 17)      # slots 1..4 slide to 2..5
    A_FLY = (17, 28)       # heart flies from reference panel to slot 1
    # Phase B: insert at position 6 (0-based slot 5)
    B_SHIFT = (31, 41)     # slot 5 (ring) slides to 6
    B_FLY = (41, 52)       # heart flies to slot 5

    def phase_t(f, span):
        s, e = span
        if f < s:
            return 0.0
        if f >= e:
            return 1.0
        return ease((f - s) / (e - s - 1))

    frames = []
    for f in range(N_FRAMES):
        canvas = background.copy()

        # --- static / sliding symbols -------------------------------------
        tA = phase_t(f, A_SHIFT)
        tB = phase_t(f, B_SHIFT)
        # heart 1 stays at slot 0
        paste(canvas, sprites[0], *slot_xy(0))
        # circle, star, triangle: slot i -> i+1 during phase A
        for i in (1, 2, 3):
            paste(canvas, sprites[i], *slot_xy(i + tA))
        # ring: slot 4 -> 5 (phase A) -> 6 (phase B)
        paste(canvas, sprites[4], *slot_xy(4 + tA + tB))

        # inserted hearts, once landed
        if f >= A_FLY[1]:
            paste(canvas, heart, *slot_xy(1))
        if f >= B_FLY[1]:
            paste(canvas, heart, *slot_xy(5))

        # restore cell borders over anything that slid across them
        canvas[border_mask] = base[border_mask]

        # --- flying hearts (drawn on top) ----------------------------------
        for span, slot in ((A_FLY, 1), (B_FLY, 5)):
            s, e = span
            if s <= f < e:
                t = ease((f - s) / (e - s))
                tx, ty = slot_xy(slot)
                x = int(round(ref_x + (tx - ref_x) * t))
                y = int(round(ref_y + (ty - ref_y) * t))
                paste(canvas, heart, x, y)

        frames.append(canvas)

    # First frame must be exactly first_frame.png
    frames[0] = base.copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp_dir, f"{i:04d}.png"))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS),
        "-i", os.path.join(tmp_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-r", str(FPS),
        OUT,
    ]
    subprocess.run(cmd, check=True)
    for name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, name))
    os.rmdir(tmp_dir)
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
