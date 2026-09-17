#!/usr/bin/env python3
"""Insert two solid diamonds at positions 1 and 2 of the symbol sequence.

Existing symbols (cells 1-5) slide right by two cells, then the two new
diamonds grow into cells 1 and 2. Everything else stays pixel-identical to
first_frame.png.
"""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 54

# Grid geometry measured from first_frame.png
X0, PITCH, CELL = 44, 105, 97          # cell k outer box: x in [X0+PITCH*(k-1), +CELL-1]
Y0, Y1 = 464, 560                      # outer rows of the boxes (border rows)
N_CELLS = 9
SHIFT = 2                              # insert 2 -> everything moves right by 2 cells
DIAMOND_CELL = 5                       # cell holding the reference diamond in the frame


def cell_x(k):  # 1-based
    return X0 + PITCH * (k - 1)


def interior(k):
    """Slice bounds (x0, x1, y0, y1) exclusive of the 1px border."""
    x = cell_x(k)
    return x + 1, x + CELL - 1, Y0 + 1, Y1


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = base.shape

    # Background = first frame with all cell interiors blanked to white.
    bg = base.copy()
    for k in range(1, N_CELLS + 1):
        a, b, c, d = interior(k)
        bg[c:d, a:b] = 255

    # Border overlay: everything in the grid band that is not interior content.
    band = np.zeros((H, W), bool)
    band[Y0:Y1 + 1, cell_x(1):cell_x(N_CELLS) + CELL] = True
    for k in range(1, N_CELLS + 1):
        a, b, c, d = interior(k)
        band[c:d, a:b] = False
    border_mask = band & (base != 255).any(2)

    # Extract symbol patches (interior content) from occupied cells.
    patches = []  # (cell index, rgb patch, alpha mask)
    for k in range(1, N_CELLS + 1):
        a, b, c, d = interior(k)
        p = base[c:d, a:b]
        m = (p != 255).any(2)
        if m.any():
            patches.append((k, p.copy(), m))

    dia_patch, dia_mask = next((p, m) for k, p, m in patches if k == DIAMOND_CELL)
    ph, pw = dia_mask.shape

    def paste(frame, patch, mask, x, y):
        h, w = mask.shape
        frame[y:y + h, x:x + w][mask] = patch[mask]

    def diamond_scaled(s):
        """Diamond patch scaled about its centre by factor s (0..1)."""
        if s >= 0.999:
            return dia_patch, dia_mask
        rgba = np.dstack([dia_patch, (dia_mask * 255).astype(np.uint8)])
        im = Image.fromarray(rgba, "RGBA")
        nw, nh = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
        small = im.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (pw, ph), (255, 255, 255, 0))
        canvas.paste(small, ((pw - nw) // 2, (ph - nh) // 2), small)
        arr = np.array(canvas)
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = (arr[..., :3] * alpha + 255 * (1 - alpha)).round().astype(np.uint8)
        return rgb, arr[..., 3] > 8

    # Timeline (frames)
    SLIDE_START, SLIDE_END = 2, 30
    D1_START, D1_END = 32, 42
    D2_START, D2_END = 42, 52

    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(base.copy())
            continue
        fr = bg.copy()
        t = ease((f - SLIDE_START) / (SLIDE_END - SLIDE_START))
        dx = int(round(t * SHIFT * PITCH))
        for k, p, m in patches:
            a, _, c, _ = interior(k)
            paste(fr, p, m, a + dx, c)
        for cell, (s0, s1) in ((1, (D1_START, D1_END)), (2, (D2_START, D2_END))):
            s = ease((f - s0) / (s1 - s0))
            if s > 0:
                p, m = diamond_scaled(s)
                a, _, c, _ = interior(cell)
                paste(fr, p, m, a, c)
        fr[border_mask] = base[border_mask]
        frames.append(fr)

    # Encode with ffmpeg (H.264, yuv420p).
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
    print(f"wrote {OUT}: {len(frames)} frames @ {FPS} fps")


if __name__ == "__main__":
    main()
