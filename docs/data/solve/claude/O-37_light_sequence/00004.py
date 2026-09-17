#!/usr/bin/env python3
"""Generate /app/output/video.mp4: turn the 3rd light on and every other light off.

The frame is reproduced from first_frame.png; only the light discs/glows change.
Each transition is a short colour crossfade (gray<->purple, glow fades in/out).
Transitions sweep left to right across the full clip duration.
"""
import os, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 35
TARGET_ON = 2            # 0-based index of the light that must end up on (3rd light)
CY = 512                 # row centre of all lights

# Geometry measured from first_frame.png (hard-edged drawing, no anti-aliasing)
# Disc (gray/purple + 2px outline) is 42px wide; glow is 51px wide.
DISC_X0 = [122, 204, 286, 368, 450, 532, 614, 696, 778, 860]   # inclusive left edge
DISC_W = 42
GLOW_W = 51
# Glow left edge: known for on-lights, inferred for off-lights from fitted
# glow centres (~143 + 81.9*i, rounded), glow bbox = centre-25 .. centre+25.
GLOW_X0 = [118, 200, 282, 364, 446, 527, 609, 692, 774, 855]
DISC_Y0, GLOW_Y0 = CY - 21, CY - 25            # 491..532 and 487..537
CELL_X_PAD = 2   # cell = glow bbox with small padding, all white outside glow

GLOW = np.array([225, 191, 231], np.float32)
ON_FILL, ON_EDGE = np.array([156, 39, 176], np.float32), np.array([123, 31, 162], np.float32)
OFF_FILL, OFF_EDGE = np.array([128, 128, 128], np.float32), np.array([64, 64, 64], np.float32)
WHITE = np.array([255, 255, 255], np.float32)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    assert base.shape == (H, W, 3)

    # Determine initial states from the frame (purple fill at disc centre => on)
    init_on = []
    for i in range(10):
        px = base[CY, DISC_X0[i] + DISC_W // 2].astype(int)
        init_on.append(bool(px[0] > 140 and px[1] < 100))
    final_on = [i == TARGET_ON for i in range(10)]

    # Build on/off templates for each light from the frame's own pixels.
    # Off light: gray disc with dark outline (pixel classes copied from an existing off light).
    # On light: glow ring + purple disc. Use pixel-class masks so colours can be lerped.
    ref_off = next(i for i in range(10) if not init_on[i])
    ref_on = next(i for i in range(10) if init_on[i])
    ys, xs = np.mgrid[0:GLOW_W, 0:GLOW_W]

    def cell_slices(i):
        x0 = GLOW_X0[i]
        return slice(GLOW_Y0, GLOW_Y0 + GLOW_W), slice(x0, x0 + GLOW_W)

    # Masks in cell coordinates (cell = glow bbox).  Disc offset inside cell:
    def disc_masks(i):
        dx = DISC_X0[i] - GLOW_X0[i]
        dy = DISC_Y0 - GLOW_Y0
        # take disc pixel classes from reference off light
        ref = base[DISC_Y0:DISC_Y0 + DISC_W, DISC_X0[ref_off]:DISC_X0[ref_off] + DISC_W]
        fill = np.all(ref == 128, axis=2)
        edge = np.all(ref == 64, axis=2)
        m_fill = np.zeros((GLOW_W, GLOW_W), bool); m_edge = m_fill.copy()
        m_fill[dy:dy + DISC_W, dx:dx + DISC_W] = fill
        m_edge[dy:dy + DISC_W, dx:dx + DISC_W] = edge
        return m_fill, m_edge

    # Glow mask from reference on light, excluding its disc (glow ring shape)
    ref_cell = base[cell_slices(ref_on)]
    m_glow_ref = np.all(ref_cell == GLOW.astype(np.uint8), axis=2) | \
        np.all(ref_cell == ON_FILL.astype(np.uint8), axis=2) | \
        np.all(ref_cell == ON_EDGE.astype(np.uint8), axis=2)   # full glow disc footprint

    def glow_offset(i):
        # +0.5 if glow centre lies right of disc centre, else -0.5
        return (GLOW_X0[i] + GLOW_W / 2) - (DISC_X0[i] + DISC_W / 2)

    def render_cell(i, t):
        """t=0 -> off, t=1 -> on."""
        m_fill, m_edge = disc_masks(i)
        m_glow = m_glow_ref if glow_offset(i) == glow_offset(ref_on) else m_glow_ref[:, ::-1]
        cell = np.full((GLOW_W, GLOW_W, 3), 255, np.float32)
        glow_col = WHITE + (GLOW - WHITE) * t
        cell[m_glow] = glow_col
        cell[m_fill] = OFF_FILL + (ON_FILL - OFF_FILL) * t
        cell[m_edge] = OFF_EDGE + (ON_EDGE - OFF_EDGE) * t
        return cell

    # Sanity: rendered states must reproduce the frame for unchanged lights
    for i in range(10):
        c = render_cell(i, 1.0 if init_on[i] else 0.0)
        ref = base[cell_slices(i)].astype(np.float32)
        if init_on[i]:
            assert np.abs(c - ref).max() < 1, f"on template mismatch light {i}"
        else:
            # off lights: glow area is white in frame; disc must match
            assert np.abs(c - ref).max() < 1, f"off template mismatch light {i}"

    # Schedule: sweep left->right over lights that change, 6-frame crossfades
    changing = [i for i in range(10) if init_on[i] != final_on[i]]
    DUR = 6
    starts = {}
    first_start, last_end = 2, N_FRAMES - 3
    span = last_end - DUR - first_start
    for k, i in enumerate(changing):
        s = first_start + (span * k) / max(1, len(changing) - 1)
        starts[i] = s

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        frame = base.copy()
        for i in changing:
            p = smoothstep((f - starts[i]) / DUR)
            if f == N_FRAMES - 1:
                p = 1.0
            t = 1.0 - p if init_on[i] else p     # fraction "on"
            cell = render_cell(i, t)
            frame[cell_slices(i)] = np.clip(np.rint(cell), 0, 255).astype(np.uint8)
        if f == 0:
            assert np.array_equal(frame, base)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
