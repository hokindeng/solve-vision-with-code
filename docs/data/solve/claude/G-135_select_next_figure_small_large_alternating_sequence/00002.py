#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: mark the correct option (3rd box, large
olive diamond) with a red circle that sweeps in over time."""
import subprocess, numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N, FPS = 60, 16
CX, CY, R, W = 626, 854, 92, 6          # circle around option 3
START, END = 8, 46                       # frames over which the arc sweeps
SS = 4                                   # supersampling for anti-aliasing

base = Image.open(BASE).convert("RGB")

def frame(i):
    if i < START:
        return base.copy()
    t = min(1.0, (i - START) / (END - START))
    t = t * t * (3 - 2 * t)              # ease in/out
    sweep = 360 * t
    layer = Image.new("RGBA", (base.width * SS, base.height * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bbox = [(CX - R) * SS, (CY - R) * SS, (CX + R) * SS, (CY + R) * SS]
    d.arc(bbox, start=-90, end=-90 + sweep, fill=(220, 30, 30, 255), width=W * SS)
    layer = layer.resize(base.size, Image.LANCZOS)
    out = base.copy().convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")

frames = [np.asarray(frame(i), dtype=np.uint8) for i in range(N)]
assert np.array_equal(frames[0], np.asarray(base))

p = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{base.width}x{base.height}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
print("wrote", OUT)
