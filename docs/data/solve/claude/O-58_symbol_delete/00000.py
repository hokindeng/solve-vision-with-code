#!/usr/bin/env python3
"""Delete the yellow solid star at position 6: fade it out, then slide
position 7 leftward into the gap. All other pixels stay as in first_frame.png."""
import os, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 45
FADE_FRAMES = 20                       # frames 0..19: star fades
SLIDE_FRAMES = N_FRAMES - FADE_FRAMES  # frames 20..44: slide left

# Measured geometry of first_frame.png
BOX_Y0, BOX_Y1 = 464, 560              # box frame rows (inclusive)
LBL_Y0, LBL_Y1 = 584, 603              # label rows
PITCH = 105
BOX6_X0, BOX6_X1 = 674, 770
BOX7_X0, BOX7_X1 = 779, 875
LBL7_X0, LBL7_X1 = 714 + PITCH, 730 + PITCH  # label "7"
BG = np.array([255, 255, 255], np.float32)

def ease(t):  # smooth ease-in-out
    return t * t * (3 - 2 * t)

def main():
    first = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
    H, W, _ = first.shape

    # Pieces that change
    star_int = first[BOX_Y0 + 3:BOX_Y1 - 2, BOX6_X0 + 3:BOX6_X1 - 2].copy()  # box 6 interior
    box7 = first[BOX_Y0:BOX_Y1 + 1, BOX7_X0:BOX7_X1 + 1].copy()             # frame + symbol
    lbl7 = first[LBL_Y0:LBL_Y1 + 1, LBL7_X0:LBL7_X1 + 1].copy()

    # Static base: everything except the moving/fading pieces
    base = first.copy()
    base[BOX_Y0 + 3:BOX_Y1 - 2, BOX6_X0 + 3:BOX6_X1 - 2] = BG
    base[BOX_Y0:BOX_Y1 + 1, BOX7_X0:BOX7_X1 + 1] = BG
    base[LBL_Y0:LBL_Y1 + 1, LBL7_X0:LBL7_X1 + 1] = BG

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        if i < FADE_FRAMES:
            a = 1.0 - i / (FADE_FRAMES - 1)          # 1 -> 0
            f[BOX_Y0 + 3:BOX_Y1 - 2, BOX6_X0 + 3:BOX6_X1 - 2] = BG + (star_int - BG) * a
            f[BOX_Y0:BOX_Y1 + 1, BOX7_X0:BOX7_X1 + 1] = box7
            f[LBL_Y0:LBL_Y1 + 1, LBL7_X0:LBL7_X1 + 1] = lbl7
        else:
            t = (i - FADE_FRAMES) / (SLIDE_FRAMES - 1)   # 0 -> 1
            dx = int(round(ease(t) * PITCH))
            x0 = BOX7_X0 - dx
            region = f[BOX_Y0:BOX_Y1 + 1, x0:x0 + box7.shape[1]]
            np.minimum(region, box7, out=region)      # darker-wins over white bg
            f[LBL_Y0:LBL_Y1 + 1, LBL7_X0:LBL7_X1 + 1] = BG + (lbl7 - BG) * (1.0 - ease(t))
        frames.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))

    assert np.array_equal(frames[0], first.astype(np.uint8))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, len(frames), "frames")

if __name__ == "__main__":
    main()
