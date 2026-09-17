#!/usr/bin/env python3
"""Generate /app/output/video.mp4: insert three hollow circles at positions 1, 5, 6.

The sequence row holds 9 slots. Starting state: [circle, diamond, triangle, square, -, -, -, -, -].
Each insertion shifts the symbols at/after the insertion index one slot to the right and
then reveals a new hollow circle (copied pixel-for-pixel from the existing one) in the gap.
Final state: [circle, circle, diamond, triangle, circle, circle, square, -, -].
Everything outside the slot interiors is left exactly as in first_frame.png.
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
N_FRAMES = 76

# Slot geometry measured from first_frame.png
SLOT_X0 = 44          # left border x of slot 1
SLOT_PITCH = 105
SLOT_SIZE = 97        # border to border inclusive
SLOT_Y0 = 464         # top border y
N_SLOTS = 9

# Interior (excluding the 1px gray border)
INT_W = SLOT_SIZE - 2
INT_H = SLOT_SIZE - 2


def slot_interior_box(i):
    """1-based slot index -> (x, y) of interior top-left."""
    x = SLOT_X0 + SLOT_PITCH * (i - 1) + 1
    y = SLOT_Y0 + 1
    return x, y


def extract_sprite(base, i):
    """Return an RGBA sprite of the symbol in slot i (None if empty)."""
    x, y = slot_interior_box(i)
    patch = base[y:y + INT_H, x:x + INT_W].astype(np.uint8)
    mask = (patch < 250).any(axis=2)
    if not mask.any():
        return None
    alpha = (mask * 255).astype(np.uint8)
    return np.dstack([patch, alpha])


def composite(frame, sprite, x, y, scale=1.0):
    """Alpha-composite sprite (RGBA ndarray) centered on the interior box at (x, y)."""
    if sprite is None:
        return
    h, w = sprite.shape[:2]
    if scale <= 0.0:
        return
    if scale != 1.0:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        im = Image.fromarray(sprite, "RGBA").resize((nw, nh), Image.LANCZOS)
        sp = np.asarray(im)
        x = x + (w - nw) / 2.0
        y = y + (h - nh) / 2.0
        h, w = nh, nw
    else:
        sp = sprite
    xi, yi = int(round(x)), int(round(y))
    x0, y0 = max(0, xi), max(0, yi)
    x1, y1 = min(W, xi + w), min(H, yi + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = sp[y0 - yi:y1 - yi, x0 - xi:x1 - xi]
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (src[:, :, :3] * a + dst * (1 - a) + 0.5).astype(np.uint8)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))

    # Background = first frame with all slot interiors cleared to white.
    bg = base.copy()
    for i in range(1, N_SLOTS + 1):
        x, y = slot_interior_box(i)
        bg[y:y + INT_H, x:x + INT_W] = 255

    sprites = {i: extract_sprite(base, i) for i in range(1, N_SLOTS + 1)}
    circle = sprites[1]
    assert circle is not None

    # Sequence as list of sprite ids (slot indices in the original frame); None = empty.
    seq = [i for i in range(1, N_SLOTS + 1) if sprites[i] is not None]  # [1,2,3,4]
    new_id = 100

    # Insertions (1-based positions in the sequence at the time of insertion).
    insertions = [1, 5, 6]

    # Timeline
    HOLD_START = 4
    SLIDE = 12
    APPEAR = 10
    STEP = SLIDE + APPEAR
    HOLD_END = N_FRAMES - HOLD_START - STEP * len(insertions)
    assert HOLD_END >= 0

    frames = []

    def render(items, moving=None, appearing=None):
        """items: list of (sprite_id, slot_pos_float). appearing: (sprite, slot_pos, scale)."""
        f = bg.copy()
        for sid, pos in items:
            sp = circle if sid >= 100 else sprites[sid]
            x, y = slot_interior_box(1)
            x = x + SLOT_PITCH * (pos - 1)
            composite(f, sp, x, y)
        if appearing is not None:
            sp, pos, sc = appearing
            x, y = slot_interior_box(1)
            x = x + SLOT_PITCH * (pos - 1)
            composite(f, sp, x, y, scale=sc)
        return f

    def static_items(s):
        return [(sid, k + 1) for k, sid in enumerate(s)]

    # Frame 0 must equal first_frame exactly.
    frames.append(base.copy())
    for _ in range(HOLD_START - 1):
        frames.append(base.copy())

    for pos in insertions:
        idx = pos - 1  # 0-based insertion index
        # Slide phase: elements at idx.. move right by one slot.
        for k in range(1, SLIDE + 1):
            t = ease(k / SLIDE)
            items = []
            for j, sid in enumerate(seq):
                p = j + 1 + (t if j >= idx else 0.0)
                items.append((sid, p))
            frames.append(render(items))
        seq.insert(idx, new_id)
        new_id += 1
        # Appear phase: new circle scales in at pos.
        others = [(sid, k + 1) for k, sid in enumerate(seq) if sid != new_id - 1]
        for k in range(1, APPEAR + 1):
            t = ease(k / APPEAR)
            if k == APPEAR:
                frames.append(render(static_items(seq)))
            else:
                frames.append(render(others, appearing=(circle, pos, t)))

    final = render(static_items(seq))
    for _ in range(HOLD_END):
        frames.append(final.copy())

    assert len(frames) == N_FRAMES, len(frames)

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "10",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")

    # Also save first/last frames for inspection.
    Image.fromarray(frames[0]).save(os.path.join(OUT_DIR, "frame_first.png"))
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "frame_last.png"))
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
