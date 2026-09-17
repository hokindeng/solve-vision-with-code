#!/usr/bin/env python3
"""Generate the 'insert two solid right triangles at positions 3 and 5' video.

The first frame is /app/first_frame.png.  The row of 9 slot boxes sits at
y=464..560 with box i (0-based) spanning x=44+105*i .. 140+105*i (1px gray
border).  Symbols live inside the 95x95 interiors.  We animate two sequential
insertions:
  1. slide diamond/triangle/outline-star (slots 3,4,5) one slot right,
     then grow a solid right triangle into slot 3;
  2. slide triangle/outline-star (slots 5,6) one slot right,
     then grow a solid right triangle into slot 5.
Everything outside the moving symbols is copied verbatim from first_frame.png.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 54
W = H = 1024

SLOT_X0 = 44        # left border x of slot 1
SLOT_STEP = 105
SLOT_Y0 = 464       # top border y
INNER = 95          # interior size (x0+1 .. x0+95)


def interior_box(slot):  # slot is 1-based
    x = SLOT_X0 + SLOT_STEP * (slot - 1) + 1
    y = SLOT_Y0 + 1
    return x, y, x + INNER, y + INNER


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    base = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)

    # Extract 95x95 interior patches (symbol on white) from occupied slots.
    def patch(slot):
        x0, y0, x1, y1 = interior_box(slot)
        return base[y0:y1, x0:x1].copy()

    # Background: the frame with slots 3..5 interiors cleared to white.
    bg = base.copy()
    for s in (3, 4, 5):
        x0, y0, x1, y1 = interior_box(s)
        bg[y0:y1, x0:x1] = 255.0

    def to_alpha(layer):
        """Split a white-background patch into (alpha, solid colour).

        Symbols are a single flat colour anti-aliased against white, so each
        pixel is  alpha*colour + (1-alpha)*255.  Recover alpha from the
        deviation from white relative to the most saturated pixel.
        """
        dev = (255.0 - layer).sum(axis=2)
        idx = np.unravel_index(np.argmax(dev), dev.shape)
        colour = layer[idx]
        alpha = np.clip(dev / max(dev[idx], 1e-6), 0.0, 1.0)
        return alpha.astype(np.float32), colour.astype(np.float32)

    def composite(frame, layer, cx, cy, scale=1.0, strength=1.0):
        """Alpha-composite a symbol patch centred at (cx, cy) over frame."""
        if strength <= 0 or scale <= 0:
            return
        alpha, colour = layer
        if scale != 1.0:
            sz = max(1, int(round(INNER * scale)))
            im = Image.fromarray((alpha * 255).astype(np.uint8)).resize((sz, sz), Image.LANCZOS)
            alpha = np.asarray(im).astype(np.float32) / 255.0
        h, w = alpha.shape
        x0 = int(round(cx - w / 2.0))
        y0 = int(round(cy - h / 2.0))
        a = (alpha * strength)[..., None]
        region = frame[y0:y0 + h, x0:x0 + w]
        region[:] = a * colour + (1.0 - a) * region

    diamond = to_alpha(patch(3))
    tri = to_alpha(patch(4))
    ostar = to_alpha(patch(5))
    new_tri = tri  # target symbol identical to existing solid right triangle

    def center(slot_pos):
        """Centre of a (possibly fractional) 1-based slot position."""
        x0 = SLOT_X0 + SLOT_STEP * (slot_pos - 1) + 1
        return x0 + INNER / 2.0, SLOT_Y0 + 1 + INNER / 2.0

    # Timeline (frame indices)
    A0, A1 = 1, 14     # slide 3,4,5 -> 4,5,6
    B0, B1 = 15, 26    # grow triangle into slot 3
    C0, C1 = 27, 39    # slide 5,6 -> 6,7
    D0, D1 = 40, 51    # grow triangle into slot 5

    frames = []
    for f in range(N_FRAMES):
        frame = bg.copy()
        # positions of the three moving symbols
        tA = ease((f - A0) / float(A1 - A0)) if f >= A0 else 0.0
        tC = ease((f - C0) / float(C1 - C0)) if f >= C0 else 0.0
        p_diamond = 3 + tA
        p_tri = 4 + tA + tC
        p_ostar = 5 + tA + tC
        for layer, p in ((diamond, p_diamond), (tri, p_tri), (ostar, p_ostar)):
            cx, cy = center(p)
            composite(frame, layer, cx, cy)
        # new triangle at slot 3
        if f >= B0:
            tB = ease((f - B0) / float(B1 - B0))
            cx, cy = center(3)
            composite(frame, new_tri, cx, cy, scale=0.2 + 0.8 * tB, strength=tB)
        # new triangle at slot 5
        if f >= D0:
            tD = ease((f - D0) / float(D1 - D0))
            cx, cy = center(5)
            composite(frame, new_tri, cx, cy, scale=0.2 + 0.8 * tD, strength=tD)
        frames.append(np.clip(np.round(frame), 0, 255).astype(np.uint8))

    # frame 0 must be exactly the source
    frames[0] = base.astype(np.uint8)

    os.makedirs(OUT_DIR, exist_ok=True)
    raw = np.stack(frames).tobytes()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
