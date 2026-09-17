"""Circle the asymmetrical shape (the scalene triangle) with a red circle,
drawn progressively as an arc sweep over 16 frames."""
import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT_DIR = "/app/output"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
N_FRAMES, FPS = 16, 16

# Target: magenta scalene triangle, bbox x 325..439, y 495..560
CX, CY, R = 382.0, 527.5, 85
WIDTH = 6
RED = (255, 0, 0)

def draw_arc(base, frac):
    """Draw the arc from 0..frac of full circle, high-res then downsample for AA."""
    if frac <= 0:
        return base.copy()
    S = 4
    w, h = base.size
    overlay = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    bbox = [(CX - R) * S, (CY - R) * S, (CX + R) * S, (CY + R) * S]
    start = -90
    end = start + 360 * frac
    if frac >= 1.0:
        d.ellipse(bbox, outline=RED + (255,), width=WIDTH * S)
    else:
        d.arc(bbox, start=start, end=end, fill=RED + (255,), width=WIDTH * S)
        # round caps
        for ang in (start, end):
            a = math.radians(ang)
            rr = R - WIDTH / 2
            px, py = (CX + rr * math.cos(a)) * S, (CY + rr * math.sin(a)) * S
            hw = WIDTH * S / 2
            d.ellipse([px - hw, py - hw, px + hw, py + hw], fill=RED + (255,))
    overlay = overlay.resize((w, h), Image.LANCZOS)
    out = base.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB")

def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    base = Image.open(BASE).convert("RGB")
    for i in range(N_FRAMES):
        # frame 0 untouched; sweep completes by the final frame (ease-in-out)
        t = i / (N_FRAMES - 1)
        frac = 0.5 - 0.5 * math.cos(math.pi * t)
        frame = draw_arc(base, frac)
        frame.save(os.path.join(FRAMES_DIR, f"frame_{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(FRAMES_DIR, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        os.path.join(OUT_DIR, "video.mp4"),
    ], check=True)

if __name__ == "__main__":
    main()
