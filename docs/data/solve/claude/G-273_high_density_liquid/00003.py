#!/usr/bin/env python3
"""Generate the buoyancy video: three identical cubes drop into three cups.

Left cup (dark liquid) is denser than the cube -> cube floats.
Middle/right cups (light liquid) are less dense -> cubes sink to the bottom.
Everything except the cubes is left exactly as in first_frame.png.
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
N_FRAMES = 80
SUBSTEPS = 20

# Measured from first_frame.png (pixel coords, inclusive)
CUBE_Y0, CUBE_Y1 = 93, 158                 # cube rows in the first frame
CUBE_X = [(235, 300), (479, 544), (723, 788)]
CUBE_H = CUBE_Y1 - CUBE_Y0 + 1              # 66 px
SURFACE_Y = [568, 479, 507]                 # first row of liquid in each cup
CUP_BOTTOM = 855                            # last interior row of every cup
REST_Y = CUP_BOTTOM - CUBE_H + 1            # cube top row when resting on bottom

# Physics (pixels, seconds). Object density normalised to 1.
G = 550.0
RHO = [1.35, 0.75, 0.75]                    # liquid density / object density
K_QUAD = [0.045, 0.016, 0.016]            # quadratic drag
C_LIN = [3.5, 0.15, 0.15]                    # linear drag (viscous, dense liquid more viscous)


def simulate():
    """Return array [N_FRAMES, 3] of cube top-row positions (float)."""
    dt = 1.0 / FPS / SUBSTEPS
    y = np.full(3, float(CUBE_Y0))
    v = np.zeros(3)
    pos = np.zeros((N_FRAMES, 3))
    settled = np.zeros(3, dtype=bool)
    for f in range(N_FRAMES):
        pos[f] = y
        for _ in range(SUBSTEPS):
            for i in range(3):
                if settled[i]:
                    continue
                bottom = y[i] + CUBE_H
                sub = np.clip((bottom - SURFACE_Y[i]) / CUBE_H, 0.0, 1.0)
                a = G * (1.0 - RHO[i] * sub) - sub * (K_QUAD[i] * v[i] * abs(v[i]) + C_LIN[i] * v[i])
                v[i] += a * dt
                y[i] += v[i] * dt
                if y[i] >= REST_Y:          # inelastic landing on the cup floor
                    y[i] = REST_Y
                    v[i] = 0.0
                    if RHO[i] < 1.0:
                        settled[i] = True
    return pos


def render():
    base = np.array(Image.open(FIRST).convert("RGB"))
    sprite = base[CUBE_Y0:CUBE_Y1 + 1, CUBE_X[0][0]:CUBE_X[0][1] + 1].copy()
    bg = base.copy()
    for x0, x1 in CUBE_X:                    # background behind the cubes is plain white
        bg[CUBE_Y0:CUBE_Y1 + 1, x0:x1 + 1] = 255

    pos = simulate()
    # Force exact first frame and a fully settled last frame.
    pos[0] = CUBE_Y0
    frames = []
    for f in range(N_FRAMES):
        img = bg.copy()
        for i, (x0, x1) in enumerate(CUBE_X):
            y = int(round(pos[f, i]))
            y = min(max(y, CUBE_Y0), REST_Y)
            spr = sprite.copy().astype(np.float32)
            # Tint the submerged part with the liquid colour so it reads as "in the liquid".
            sy = SURFACE_Y[i]
            if y + CUBE_H > sy:
                k0 = max(0, sy - y)
                liquid = base[sy + 3, x0 + 3].astype(np.float32)  # fill colour below surface line
                spr[k0:] = spr[k0:] * 0.72 + liquid * 0.28
            img[y:y + CUBE_H, x0:x1 + 1] = np.clip(spr, 0, 255).astype(np.uint8)
        frames.append(img)
    frames[0] = base.copy()
    return frames


def write_video(frames):
    os.makedirs(OUT_DIR, exist_ok=True)
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    frs = render()
    write_video(frs)
    print("wrote", OUT, len(frs), "frames")
