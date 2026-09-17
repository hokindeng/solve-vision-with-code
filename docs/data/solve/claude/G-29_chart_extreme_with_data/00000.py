"""Highlight the maximum-value pie segment with an animated red outline."""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
W = H = 1024
FPS, N = 16, 48
SS = 4  # supersampling for anti-aliased red stroke

# Pie geometry (measured from first_frame.png; wedge order = matplotlib startangle=90, clockwise)
CX, CY, R = 511.5, 541.5, 389.5
values = [4.7, 21.8, 41.9, 5.1, 26.5]
imax = int(np.argmax(values))
start = 90.0
angles = []
for v in values:
    span = 360.0 * v / sum(values)
    angles.append((start, start - span))
    start -= span
a0, a1 = angles[imax]  # a0 > a1 (clockwise sweep), in degrees, math orientation (CCW positive)


def pt(ang_deg, rad):
    t = math.radians(ang_deg)
    return (CX + rad * math.cos(t), CY - rad * math.sin(t))


# Build the outline path: center -> arc start -> along arc -> arc end -> center
arc_pts = [pt(a0 - (a0 - a1) * k / 400.0, R) for k in range(401)]
path = [(CX, CY)] + arc_pts + [(CX, CY)]
seg_len = [math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)]
total = sum(seg_len)


def partial_path(frac):
    """Return the points of the path up to fraction `frac` of its total length."""
    target = frac * total
    out = [path[0]]
    acc = 0.0
    for i, L in enumerate(seg_len):
        if acc + L >= target:
            t = (target - acc) / L if L else 0
            p, q = path[i], path[i + 1]
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
            break
        out.append(path[i + 1])
        acc += L
    return out


base = Image.open(SRC).convert("RGB")
WIDTH = 7  # stroke width in output pixels
RED = (230, 0, 0, 255)

frames = []
draw_frames = N - 4  # finish slightly before the end and hold
for f in range(N):
    frac = min(1.0, f / (draw_frames - 1)) if f > 0 else 0.0
    frame = base.copy()
    if frac > 0:
        layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pts = [(x * SS, y * SS) for x, y in partial_path(frac)]
        if len(pts) >= 2:
            d.line(pts, fill=RED, width=WIDTH * SS, joint="curve")
            rr = WIDTH * SS / 2
            for (x, y) in (pts[0], pts[-1]):
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=RED)
        layer = layer.resize((W, H), Image.LANCZOS)
        frame = Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")
    frames.append(np.asarray(frame))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for fr in frames:
    proc.stdin.write(fr.tobytes())
proc.stdin.close()
proc.wait()
Image.fromarray(frames[-1]).save(os.path.join(HERE, "output", "last_frame.png"))
print("wrote", OUT, "max segment:", values[imax], "angles", a0, a1)
