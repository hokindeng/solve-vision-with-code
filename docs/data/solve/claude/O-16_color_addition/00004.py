#!/usr/bin/env python3
"""Two balls move toward each other at equal speed and merge at the midpoint,
mixing their colors additively (light mixing) wherever they overlap."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

FIRST = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 80

# Ball geometry recovered from first_frame.png (exact PIL reproduction verified):
# ImageDraw.ellipse([cx-R, cy-R, cx+R, cy+R], fill=color, outline=black, width=OUTLINE)
R = 120
OUTLINE = 2
BALLS = [((300, 406), (179, 70, 59)),   # red-ish ball
         ((733, 594), (75, 138, 77))]   # green-ish ball
BLACK = (0, 0, 0)


def masks(center):
    """Return (interior, ring) boolean masks for a ball at integer center."""
    cx, cy = center
    disc = Image.new("L", (W, H), 0)
    ImageDraw.Draw(disc).ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)
    inner = Image.new("L", (W, H), 0)
    ImageDraw.Draw(inner).ellipse([cx - R, cy - R, cx + R, cy + R], fill=255,
                                  outline=0, width=OUTLINE)
    disc = np.array(disc) > 0
    interior = np.array(inner) > 0
    return interior, disc & ~interior


def render(bg, centers):
    frame = bg.copy()
    (i1, o1), (i2, o2) = masks(centers[0]), masks(centers[1])
    c1 = np.array(BALLS[0][1], np.int32)
    c2 = np.array(BALLS[1][1], np.int32)
    frame[i1 & ~i2] = c1
    frame[i2 & ~i1] = c2
    frame[i1 & i2] = np.clip(c1 + c2, 0, 255)          # additive light mixing
    frame[(o1 & ~i2) | (o2 & ~i1)] = BLACK             # outline of the union
    return frame


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    # Background = first frame with the balls erased (they sit on flat white).
    bg = first.copy()
    for c, _ in BALLS:
        i, o = masks(c)
        bg[i | o] = 255

    (x1, y1), (x2, y2) = BALLS[0][0], BALLS[1][0]
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)

    for k in range(N_FRAMES):
        t = k / (N_FRAMES - 1)  # constant speed, fully merged in the final frame
        p1 = (int(round(x1 + (mx - x1) * t)), int(round(y1 + (my - y1) * t)))
        p2 = (int(round(x2 + (mx - x2) * t)), int(round(y2 + (my - y2) * t)))
        frame = render(bg, (p1, p2)) if k else first
        if k == 0:
            assert (render(bg, (p1, p2)) == first).all(), "frame 0 must reproduce first_frame.png"
        if k == N_FRAMES - 1:
            assert p1 == p2, "balls must coincide in the last frame"
            Image.fromarray(frame).save("/app/output/last_frame.png")
        ff.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    assert ff.returncode == 0
    print("wrote", OUT)


if __name__ == "__main__":
    main()
