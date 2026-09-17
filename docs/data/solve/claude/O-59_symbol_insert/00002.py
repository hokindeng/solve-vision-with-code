#!/usr/bin/env python3
"""Insert a yellow solid diamond at position 6: slots 6-7 slide right, then the
new symbol fades in above the gap and slides down into slot 6."""
import subprocess, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS = 32, 16

# Geometry measured from first_frame.png
CELL_X0 = [96, 201, 306, 411, 516, 621, 726, 831]   # left border column of each cell
CELL_W, CELL_Y0, CELL_Y1 = 97, 464, 560              # border inclusive
PITCH = 105
TARGET = 6                                          # 1-based slot
REF_BOX = (907, 38, 985, 116)                       # yellow diamond in reference panel (x0,y0,x1,y1)

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

def interior(img, i):
    x0 = CELL_X0[i] + 1
    return img[CELL_Y0 + 1:CELL_Y1, x0:x0 + CELL_W - 2].copy()

def paste(dst, patch, mask, x, y):
    h, w = mask.shape
    dst[y:y + h, x:x + w][mask] = patch[mask]

def blend(dst, patch, alpha, x, y):
    h, w = alpha.shape
    reg = dst[y:y + h, x:x + w].astype(np.float32)
    a = alpha[..., None]
    dst[y:y + h, x:x + w] = (reg * (1 - a) + patch.astype(np.float32) * a).round().astype(np.uint8)

def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    ti = TARGET - 1
    # Symbols that must move: those in slots >= target (slot 8 is empty).
    movers = []
    for i in range(ti, len(CELL_X0) - 1):
        p = interior(base, i)
        m = (p != 255).any(axis=2)
        if m.any():
            movers.append((i, p, m))
    bg = base.copy()
    for i, p, m in movers:
        x0 = CELL_X0[i] + 1
        bg[CELL_Y0 + 1:CELL_Y1, x0:x0 + CELL_W - 2][m] = 255

    # New symbol: copy of the reference panel diamond, centred in the target cell.
    rx0, ry0, rx1, ry1 = REF_BOX
    sym = base[ry0:ry1, rx0:rx1].copy()
    sym_mask = ((sym != 255).any(axis=2)).astype(np.float32)
    sh, sw = sym_mask.shape
    cx = CELL_X0[ti] + CELL_W // 2
    cy = (CELL_Y0 + CELL_Y1) // 2
    fx, fy = cx - sw // 2, cy - sh // 2

    # Timeline (frame indices)
    slide_s, slide_e = 1, 15
    drop_s, drop_e = 15, 31
    rise = 90  # pixels above the slot where the symbol appears

    frames = []
    for f in range(N_FRAMES):
        img = bg.copy()
        ts = ease((f - slide_s) / (slide_e - slide_s))
        dx = int(round(ts * PITCH))
        for i, p, m in movers:
            paste(img, p, m, CELL_X0[i] + 1 + dx, CELL_Y0 + 1)
        if f >= drop_s:
            td = (f - drop_s) / (drop_e - drop_s)
            alpha = ease(min(td / 0.5, 1.0))          # fade in during first half
            y = int(round(fy - rise * (1 - ease(td))))
            blend(img, sym, sym_mask * alpha, fx, y)
        frames.append(img)
    frames[0] = base.copy()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)

if __name__ == "__main__":
    main()
