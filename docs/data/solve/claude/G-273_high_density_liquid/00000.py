#!/usr/bin/env python3
"""Generate the buoyancy video: 4 identical objects fall into 4 cups.

Cups 1-3 hold a light (low-density) liquid -> the object sinks to the bottom.
Cup 4 holds the darker, denser liquid -> the object floats at the surface.

Everything except the moving objects (and the liquid pixels they cover) is
copied verbatim from first_frame.png.
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
W = H = 1024

# --- geometry measured from first_frame.png -------------------------------
# object sprite: ellipse bbox incl. outline (inclusive) x 125..201, y 102..150 -> 77 x 49
OBJ_W, OBJ_H = 77, 49
OBJ_HALF_H = OBJ_H / 2.0
OBJ_CX = [163, 407, 651, 895]        # object centres (x)
OBJ_CY0 = 126                        # object centre (y) at frame 0
# cups: frame x0..x1 (1px dark outline), y 191..856
CUP_X = [(61, 265), (305, 509), (549, 753), (793, 997)]
CUP_TOP, CUP_BOT = 191, 856          # outline rows
INNER_BOT = CUP_BOT - 1              # last interior row (855)
# liquid surface top row (2px highlight line, then fill below)
SURF_Y = [522, 491, 528, 611]
DENSE = [False, False, False, True]  # only cup 4 liquid is denser than object

FLOAT_FRACTION = 0.62   # fraction of the object volume below the surface when floating
TINT_ALPHA = 0.40       # how strongly the liquid tints a submerged object


def submerged_fraction(depth_of_center):
    """Area fraction of a unit-ish ellipse (half-height b) lying below a
    horizontal line located `depth_of_center` above the ellipse centre
    (positive = centre is below the surface)."""
    b = OBJ_HALF_H
    d = np.clip(depth_of_center / b, -1, 1)
    # fraction of ellipse below a line at signed distance -d from centre
    # (same as for a circle after affine scaling)
    return 0.5 + (d * np.sqrt(1 - d * d) + np.arcsin(d)) / np.pi


def float_equilibrium_offset():
    """Centre depth below the surface such that FLOAT_FRACTION is submerged."""
    ds = np.linspace(-OBJ_HALF_H, OBJ_HALF_H, 20001)
    fr = submerged_fraction(ds)
    return float(ds[np.argmin(np.abs(fr - FLOAT_FRACTION))])


def trajectories():
    """Return array [N_FRAMES, 4] of object centre y positions (float)."""
    g = 1.05                 # px / frame^2 in air
    v_term = 7.0             # terminal sinking speed in liquid (px / frame)
    tau_sink = 3.0           # frames for entry speed to decay toward v_term
    y_eq_off = float_equilibrium_offset()

    ys = np.zeros((N_FRAMES, 4))
    for i in range(4):
        y = float(OBJ_CY0)
        v = 0.0
        in_liquid = False
        t_entry = 0
        v_entry = 0.0
        surf = SURF_Y[i]
        bottom_center = INNER_BOT - OBJ_HALF_H + 0.5   # resting on the cup floor
        y_eq = surf + y_eq_off                          # floating equilibrium
        for t in range(N_FRAMES):
            ys[t, i] = y
            if not in_liquid:
                v += g
                y += v
                if y + OBJ_HALF_H * 0.3 >= surf:   # object nose touches liquid
                    in_liquid = True
                    t_entry = t
                    v_entry = v
            else:
                k = t - t_entry
                if not DENSE[i]:
                    # sinks: speed relaxes exponentially to terminal speed
                    v = v_term + (v_entry - v_term) * np.exp(-k / tau_sink)
                    y = min(y + v, bottom_center)
                else:
                    # floats: damped oscillation about the equilibrium depth
                    omega = 0.42
                    zeta = 0.55
                    acc = -omega ** 2 * (y - y_eq) - 2 * zeta * omega * v
                    v += acc
                    y += v
                    y = min(y, bottom_center)
        ys[:, i] = np.minimum(ys[:, i], bottom_center)
    return ys


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = np.array(Image.open(FIRST).convert("RGB"))

    # sprite (object) cut from frame 0; mask = non-white pixels in its bbox
    x0, y0 = OBJ_CX[0] - OBJ_W // 2, OBJ_CY0 - OBJ_H // 2
    sprite = first[y0:y0 + OBJ_H, x0:x0 + OBJ_W].copy()
    smask = (sprite != 255).any(axis=2)

    # background with the objects erased (they sit on pure white)
    bg = first.copy()
    for cx in OBJ_CX:
        bx0 = cx - OBJ_W // 2
        bg[y0:y0 + OBJ_H, bx0:bx0 + OBJ_W][smask] = 255

    # liquid mask (interior of each cup at/below its surface line)
    liquid = np.zeros((H, W), dtype=bool)
    for (cx0, cx1), sy in zip(CUP_X, SURF_Y):
        liquid[sy:INNER_BOT + 1, cx0 + 1:cx1] = True

    ys = trajectories()
    frames = []
    for t in range(N_FRAMES):
        if t == 0:
            frames.append(first.copy())
            continue
        f = bg.copy()
        for i, cx in enumerate(OBJ_CX):
            cy = int(round(ys[t, i]))
            ox, oy = cx - OBJ_W // 2, cy - OBJ_H // 2
            region = f[oy:oy + OBJ_H, ox:ox + OBJ_W]
            liq = liquid[oy:oy + OBJ_H, ox:ox + OBJ_W]
            orig = bg[oy:oy + OBJ_H, ox:ox + OBJ_W]
            # draw the object
            region[smask] = sprite[smask]
            # tint the part that is under the liquid with the liquid colour
            sub = smask & liq
            blend = (region[sub].astype(np.float32) * (1 - TINT_ALPHA)
                     + orig[sub].astype(np.float32) * TINT_ALPHA)
            region[sub] = np.clip(blend + 0.5, 0, 255).astype(np.uint8)
        frames.append(f)

    # encode with ffmpeg (H.264, yuv420p)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-movflags", "+faststart", OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    # also dump the last frame for quick inspection
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
