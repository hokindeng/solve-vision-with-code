"""Generate a video that circles the Chinese character (道) in first_frame.png.

The red circle is drawn progressively as an arc that sweeps around the
character over the full 3-second duration. Every pixel outside the circle
stroke is left identical to first_frame.png.
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 48
W = H = 1024

# Bounding box of 道 (measured from the frame): x 603..727, y 350..472
CX, CY = 665.0, 411.0
RADIUS = 92.0          # comfortably encloses the ~125x123 px glyph
STROKE = 7
RED = (220, 30, 30)
SS = 4                 # supersampling factor for anti-aliased stroke


def ease(t: float) -> float:
    """Smooth ease-in-out for pen motion."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def draw_arc_layer(sweep_deg: float):
    """Return an RGBA layer with the partial circle drawn (anti-aliased)."""
    big = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    if sweep_deg <= 0:
        return big.resize((W, H), Image.LANCZOS)
    start = -100.0  # begin near top-left, like a hand drawing
    r = RADIUS * SS
    box = [CX * SS - r, CY * SS - r, CX * SS + r, CY * SS + r]
    w = STROKE * SS
    d.arc(box, start=start, end=start + min(sweep_deg, 359.999),
          fill=RED + (255,), width=w)
    # round caps at both ends of the arc
    for ang in (start, start + min(sweep_deg, 360.0)):
        a = math.radians(ang)
        px, py = CX * SS + r * math.cos(a), CY * SS + r * math.sin(a)
        d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2],
                  fill=RED + (255,))
    return big.resize((W, H), Image.LANCZOS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    frames = []
    # frame 0: untouched first frame; circle completes at ~frame 40 and holds
    draw_frames = 40
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(base.copy())
            continue
        t = min(1.0, i / draw_frames)
        sweep = 360.0 * ease(t)
        layer = draw_arc_layer(sweep)
        fr = base.copy()
        fr.paste(layer, (0, 0), layer)
        frames.append(fr)

    # verify frame 0 is identical to first frame
    assert np.array_equal(np.asarray(frames[0]), np.asarray(base))

    raw = b"".join(np.asarray(f, dtype=np.uint8).tobytes() for f in frames)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "8",
        "-preset", "medium", "-movflags", "+faststart",
        OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    print(f"wrote {OUT} ({N_FRAMES} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
