#!/usr/bin/env python3
"""Animate a red circle being drawn around the Chinese character (道) in first_frame.png."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES = 16, 48
# Bounding box of 道 (measured from pixels): x 442..566, y 112..234
CX, CY, R, WIDTH = 504.0, 173.0, 92.0, 7
SS = 4  # supersampling factor for a smooth antialiased stroke

def circle_layer(frac):
    """Return an RGBA overlay of the arc drawn from 0..frac of the full circle."""
    size = 1024 * SS
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if frac <= 0:
        return layer.resize((1024, 1024), Image.LANCZOS)
    d = ImageDraw.Draw(layer)
    bbox = [(CX - R) * SS, (CY - R) * SS, (CX + R) * SS, (CY + R) * SS]
    start = -90  # begin at top
    end = start + 360 * min(frac, 1.0)
    if frac >= 1.0:
        d.ellipse(bbox, outline=(220, 20, 20, 255), width=WIDTH * SS)
    else:
        d.arc(bbox, start, end, fill=(220, 20, 20, 255), width=WIDTH * SS)
        # round caps so the pen looks natural
        for ang in (start, end):
            a = np.deg2rad(ang)
            px, py = (CX + R * np.cos(a)) * SS, (CY + R * np.sin(a)) * SS
            r = WIDTH * SS / 2 - SS  # inside the stroke edges
            d.ellipse([px - r, py - r, px + r, py + r], fill=(220, 20, 20, 255))
    return layer.resize((1024, 1024), Image.LANCZOS)

def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep

def main():
    base = Image.open(BASE).convert("RGB")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    draw_start, draw_end = 4, 40  # frames: hold, draw, hold
    with tempfile.TemporaryDirectory() as td:
        for i in range(N_FRAMES):
            if i < draw_start:
                frac = 0.0
            elif i >= draw_end:
                frac = 1.0
            else:
                frac = ease((i - draw_start) / (draw_end - draw_start))
            frame = base.copy()
            if frac > 0:
                frame.paste(Image.new("RGB", frame.size, (220, 20, 20)), mask=circle_layer(frac).split()[3])
            frame.save(os.path.join(td, f"f{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
            "-r", str(FPS), OUT], check=True)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
