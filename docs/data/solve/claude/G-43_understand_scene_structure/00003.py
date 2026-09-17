"""Draw a green rectangular box around the bathroom (top-right room) of the floorplan.

The box is traced progressively along its perimeter over 28 frames at 16 fps.
Frame 0 is the untouched first frame; the last frame shows the complete box.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 28

# Bathroom walls (measured): left 693-702, right 960-969, top 56-65, bottom 305-315.
# The box runs along the wall centerlines so it hugs the room.
X0, Y0, X1, Y1 = 697, 60, 964, 310
THICK = 8
GREEN = (0, 200, 0)

base = Image.open(SRC).convert("RGB")


def perimeter_points(t):
    """Return polyline points covering fraction t (0..1) of the rectangle perimeter, clockwise from top-left."""
    corners = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1), (X0, Y0)]
    lengths = [abs(corners[i + 1][0] - corners[i][0]) + abs(corners[i + 1][1] - corners[i][1]) for i in range(4)]
    total = sum(lengths)
    remaining = t * total
    pts = [corners[0]]
    for i in range(4):
        a, b = corners[i], corners[i + 1]
        if remaining >= lengths[i]:
            pts.append(b)
            remaining -= lengths[i]
        else:
            f = remaining / lengths[i]
            pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            break
    return pts


def render(i):
    img = base.copy()
    if i == 0:
        return img
    t = i / (N_FRAMES - 1)
    # ease-in-out for smoother pacing
    t = 0.5 - 0.5 * np.cos(np.pi * t)
    d = ImageDraw.Draw(img)
    if t >= 1.0:
        d.rectangle([X0 - THICK // 2, Y0 - THICK // 2, X1 + THICK // 2, Y1 + THICK // 2], outline=GREEN, width=THICK)
    else:
        pts = perimeter_points(t)
        d.line(pts, fill=GREEN, width=THICK, joint="curve")
        # square caps at the start so the origin corner is fully covered
        d.rectangle([X0 - THICK // 2, Y0 - THICK // 2, X0 + THICK // 2, Y0 + THICK // 2], fill=GREEN)
    return img


frames = [np.array(render(i)) for i in range(N_FRAMES)]

ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", "1024x1024", "-r", str(FPS), "-i", "-",
     "-sws_flags", "full_chroma_int+accurate_rnd+bitexact",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow", OUT],
    stdin=subprocess.PIPE,
)
for f in frames:
    ffmpeg.stdin.write(f.tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()
print("wrote", OUT)
