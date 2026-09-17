#!/usr/bin/env python3
"""Generate the analogy-completion video for the shape/color-then-scale task.

Bottom row: the blue (color_316) large minus is transformed in two steps.
  step 1 (slot 2): same size, color changes to olive (color_118)
  step 2 (slot 3): olive color, size shrinks from large to small
Everything else in the frame is kept exactly as in first_frame.png.
"""
import subprocess, os
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES, FPS = 60, 16
COLOR_A = np.array([76, 89, 153], float)    # color_316 (blue)
COLOR_B = np.array([153, 153, 30], float)   # color_118 (olive)
OUTLINE = (0, 0, 0)

# geometry measured from first_frame.png
MINUS_BBOX = (110, 665, 250, 699)            # x0,y0,x1,y1 inclusive, large minus incl. 2px outline
Q_BOXES = {2: (500, 655, 537, 708), 3: (836, 655, 873, 708)}   # regions containing the '?' marks
SLOT_CX = {2: 518, 3: 854}
CY = 682
LARGE_W, LARGE_H = MINUS_BBOX[2] - MINUS_BBOX[0] + 1, MINUS_BBOX[3] - MINUS_BBOX[1] + 1
SMALL_SCALE = 0.5


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def draw_minus(img, cx, cy, scale, fill, alpha=1.0):
    """Draw an outlined minus (rectangle) centred at (cx, cy) with given scale, supersampled."""
    ss = 4
    w, h = LARGE_W * scale, LARGE_H * scale
    ow = max(1.0, 2.0 * scale)
    pad = 4
    W, H = int(w) + 2 * pad, int(h) + 2 * pad
    layer = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0 = (W / 2 - w / 2) * ss
    y0 = (H / 2 - h / 2) * ss
    x1 = (W / 2 + w / 2) * ss - 1
    y1 = (H / 2 + h / 2) * ss - 1
    d.rectangle([x0, y0, x1, y1], fill=tuple(int(round(c)) for c in fill) + (255,))
    o = ow * ss
    d.rectangle([x0, y0, x1, y1], outline=OUTLINE + (255,), width=int(round(o)))
    layer = layer.resize((W, H), Image.LANCZOS)
    if alpha < 1.0:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer.putalpha(a)
    px, py = int(round(cx - W / 2)), int(round(cy - H / 2))
    img.paste(layer, (px, py), layer)


def main():
    base = Image.open(FIRST).convert("RGB")
    base_np = np.array(base).astype(float)
    white = np.full_like(base_np, 255.0)
    frames = []
    for i in range(N_FRAMES):
        arr = base_np.copy()

        # --- phase 1: slot 2 -----------------------------------------------
        q_fade2 = ease((i - 0) / 7.0)          # frames 0..7: '?' fades out
        x0, y0, x1, y1 = Q_BOXES[2]
        arr[y0:y1 + 1, x0:x1 + 1] = (1 - q_fade2) * arr[y0:y1 + 1, x0:x1 + 1] + q_fade2 * white[y0:y1 + 1, x0:x1 + 1]

        # --- phase 2: slot 3 -----------------------------------------------
        q_fade3 = ease((i - 30) / 7.0)         # frames 30..37: '?' fades out
        x0, y0, x1, y1 = Q_BOXES[3]
        arr[y0:y1 + 1, x0:x1 + 1] = (1 - q_fade3) * arr[y0:y1 + 1, x0:x1 + 1] + q_fade3 * white[y0:y1 + 1, x0:x1 + 1]

        img = Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))

        # slot 2: blue minus fades in (8..13), then recolours to olive (13..29)
        if i >= 8:
            a2 = ease((i - 8) / 5.0)
            c_t = ease((i - 13) / 16.0)
            col = COLOR_A * (1 - c_t) + COLOR_B * c_t
            draw_minus(img, SLOT_CX[2], CY, 1.0, col, alpha=a2)

        # slot 3: olive minus fades in (38..43), then shrinks to small (43..57)
        if i >= 38:
            a3 = ease((i - 38) / 5.0)
            s_t = ease((i - 43) / 14.0)
            scale = 1.0 * (1 - s_t) + SMALL_SCALE * s_t
            draw_minus(img, SLOT_CX[3], CY, scale, COLOR_B, alpha=a3)

        frames.append(np.array(img))

    frames[0] = np.array(base)  # first frame is exactly the source image
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for k, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp_dir, f"{k:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fn))
    os.rmdir(tmp_dir)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
