#!/usr/bin/env python3
"""Generate the buoyancy video: 4 identical objects drop into 4 cups.
Cups 1-3 hold a dense (dark) liquid -> objects float.
Cup 4 holds a light (pale green) liquid -> object sinks."""
import subprocess, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 80, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# --- measured geometry from first_frame.png -------------------------------
OBJ_TOP, OBJ_SIZE = 94, 67                      # object rows 94..160 incl. 1px border
OBJ_LEFTS = [130, 374, 618, 862]                # object cols (67 wide)
CUP_INNER_BOTTOM = 856                          # last interior row of every cup
SURFACE_TOP = [563, 614, 596, 498]              # 2px surface line starts here; liquid fill below
OBJ_FILL = np.array([243, 156, 18], np.uint8)
OBJ_EDGE = np.array([211, 84, 0], np.uint8)

# liquid colours sampled from the frame (used for the submerged tint)
LIQ_COLOR = [tuple(base[s + 10, l + 30]) for s, l in zip(SURFACE_TOP, OBJ_LEFTS)]

# --- physics ------------------------------------------------------------
RHO_OBJ = 1.0
RHO_LIQ = [1.43, 1.43, 1.43, 0.60]              # >1 floats, <1 sinks
G = 1.16                                        # px / frame^2
DRAG = [0.30, 0.30, 0.30, 0.00]                 # linear drag when submerged
DRAG2 = [0.05, 0.05, 0.05, 0.006]               # quadratic drag when submerged
SUBSTEPS = 50


def simulate(i):
    """Return the object's top-row y for each frame."""
    y, v = float(OBJ_TOP), 0.0
    surf = SURFACE_TOP[i] + 2                   # first liquid row below the surface line
    ys = []
    dt = 1.0 / SUBSTEPS
    for _ in range(N_FRAMES):
        ys.append(y)
        for _ in range(SUBSTEPS):
            bottom = y + OBJ_SIZE
            f = np.clip((bottom - surf) / OBJ_SIZE, 0.0, 1.0)   # submerged fraction
            a = G * (1.0 - RHO_LIQ[i] * f / RHO_OBJ) - (DRAG[i] * v + DRAG2[i] * v * abs(v)) * f
            v += a * dt
            y += v * dt
            if y + OBJ_SIZE > CUP_INNER_BOTTOM + 1:               # rest on cup floor
                y = CUP_INNER_BOTTOM + 1 - OBJ_SIZE
                v = 0.0
    return ys


# background with the objects erased (they sit on pure white)
bg = base.copy()
for l in OBJ_LEFTS:
    bg[OBJ_TOP:OBJ_TOP + OBJ_SIZE, l:l + OBJ_SIZE] = 255

trajs = [simulate(i) for i in range(4)]


def render(k):
    img = bg.copy()
    for i, l in enumerate(OBJ_LEFTS):
        y0 = int(round(trajs[i][k]))
        y1 = y0 + OBJ_SIZE
        # draw the object
        img[y0:y1, l:l + OBJ_SIZE] = OBJ_FILL
        img[y0, l:l + OBJ_SIZE] = OBJ_EDGE
        img[y1 - 1, l:l + OBJ_SIZE] = OBJ_EDGE
        img[y0:y1, l] = OBJ_EDGE
        img[y0:y1, l + OBJ_SIZE - 1] = OBJ_EDGE
        # submerged part: tint with the liquid colour so it reads as under the surface
        s = SURFACE_TOP[i]
        sub0 = max(y0, s + 2)
        if sub0 < y1:
            liq = np.array(LIQ_COLOR[i], np.float32)
            region = img[sub0:y1, l:l + OBJ_SIZE].astype(np.float32)
            img[sub0:y1, l:l + OBJ_SIZE] = (0.65 * region + 0.35 * liq).astype(np.uint8)
        # keep the surface line (meniscus) on top
        if y0 < s + 2 and y1 > s:
            img[s:s + 2, l:l + OBJ_SIZE] = base[s:s + 2, l:l + OBJ_SIZE]
    return img


os.makedirs(OUT_DIR, exist_ok=True)
frames = [render(k) for k in range(N_FRAMES)]
assert np.array_equal(frames[0], base), "first frame must match first_frame.png"

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for f in frames:
    ff.stdin.write(f.tobytes())
ff.stdin.close()
ff.wait()
print("wrote", OUT)
if __name__ == "__main__" and os.environ.get("DEBUG_TRAJ"):
    for i, t in enumerate(trajs):
        print(i, [round(t[k]) for k in range(0, N_FRAMES, 4)])
