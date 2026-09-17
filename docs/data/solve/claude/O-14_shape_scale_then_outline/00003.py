"""Generate /app/output/video.mp4: the bottom-left square shrinks (scale 0.8, matching
the worked example in the top row), then its thick stroke thins to a 2 px outline.
Everything else in the frame stays identical to first_frame.png."""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS, SS = 16, 16, 8  # SS = supersampling factor for antialiasing

COLOR = (175, 191, 95)
CX, CY = 173.5, 682.0          # centre of the square (outer bbox x 119..227, y 628..736)
OUTER_HALF0 = 54.5             # outer half-size of the original square
STROKE0, STROKE1 = 6.0, 2.0    # stroke widths before / after the outline conversion
SCALE = 0.8                    # 109 px -> 87 px, same ratio as the example diamond
# Region we are allowed to repaint (white apart from the square in the source frame)
RX0, RY0, RX1, RY1 = 100, 610, 250, 755


def ease(t):
    return t * t * (3 - 2 * t)


def draw_square(bg, outer_half, stroke):
    """Return a copy of bg with the square region repainted with the given geometry."""
    w, h = (RX1 - RX0) * SS, (RY1 - RY0) * SS
    hi = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(hi)
    cx, cy = (CX - RX0) * SS, (CY - RY0) * SS
    inner_half = outer_half - stroke
    o, i = outer_half * SS, inner_half * SS
    d.rectangle([cx - o, cy - o, cx + o - 1, cy + o - 1], fill=COLOR)
    d.rectangle([cx - i, cy - i, cx + i - 1, cy + i - 1], fill=(255, 255, 255))
    lo = hi.resize((RX1 - RX0, RY1 - RY0), Image.BOX)
    out = bg.copy()
    out.paste(lo, (RX0, RY0))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(BASE).convert("RGB")
    frames = [base]
    half_a = 8  # frames 0..8: scale; frames 8..15: fill -> outline
    for f in range(1, N_FRAMES):
        if f <= half_a:
            t = ease(f / half_a)
            outer_half = OUTER_HALF0 * (1 - t) + OUTER_HALF0 * SCALE * t
            stroke = STROKE0 * outer_half / OUTER_HALF0  # stroke scales with the shape
        else:
            t = ease((f - half_a) / (N_FRAMES - 1 - half_a))
            outer_half = OUTER_HALF0 * SCALE
            s_mid = STROKE0 * SCALE
            stroke = s_mid * (1 - t) + STROKE1 * t
        frames.append(draw_square(base, outer_half, stroke))

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for k, fr in enumerate(frames):
        fr.save(os.path.join(tmp, f"{k:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
