"""Generate the analogy-completion video for the crescent/minus task.

Top row (given): green crescent -> orange crescent -> orange crescent moved up.
Bottom row: the minus is (1) recolored green -> orange in the middle slot,
then (2) the orange minus is moved up in the right slot.  Everything else in
first_frame.png stays untouched.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = Image.open(os.path.join(HERE, "first_frame.png")).convert("RGB")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, N = 16, 60

# --- measured from first_frame.png -------------------------------------------
GREEN = (210, 229, 114)
ORANGE = (229, 168, 45)
OUTLINE = (0, 0, 0)
# minus rectangle: outer bbox incl. 1px outline
MX0, MY0, MX1, MY1 = 116, 662, 274, 701
MW, MH = MX1 - MX0, MY1 - MY0
COL_STEP = 287.5          # horizontal distance between the three slots
UP_SHIFT = 60             # crescent C is 60 px above crescent B
Q_MID = (462, 654, 503, 710)    # bbox to blank out the middle "?"
Q_RIGHT = (749, 654, 790, 710)  # bbox to blank out the right "?"


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp_col(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def draw_minus(img, cx, cy, fill, alpha=1.0):
    """Draw the minus (outlined rectangle) centred at (cx, cy) with opacity alpha."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0 = int(round(cx - MW / 2))
    y0 = int(round(cy - MH / 2))
    d.rectangle([x0, y0, x0 + MW, y0 + MH], fill=fill + (255,), outline=OUTLINE + (255,))
    if alpha < 1.0:
        a = layer.split()[3].point(lambda v: int(v * alpha))
        layer.putalpha(a)
    img.paste(layer, (0, 0), layer)


def fade_box(img, box, alpha):
    """Blend the region `box` toward white by `alpha` (1 = fully white)."""
    if alpha <= 0:
        return
    region = img.crop(box)
    white = Image.new("RGB", region.size, (255, 255, 255))
    img.paste(Image.blend(region, white, min(alpha, 1.0)), box[:2])


def frame(i):
    img = BASE.copy()
    mid_cx = (MX0 + MX1) / 2 + COL_STEP
    right_cx = (MX0 + MX1) / 2 + 2 * COL_STEP
    cy0 = (MY0 + MY1) / 2

    # ---- step 1 (frames 4-28): middle "?" -> minus, recolored green -> orange
    t_in = smooth((i - 4) / 6.0)          # "?" fades out / minus fades in
    t_col = smooth((i - 9) / 19.0)        # color transition
    if i >= 4:
        fade_box(img, Q_MID, t_in)
        draw_minus(img, mid_cx, cy0, lerp_col(GREEN, ORANGE, t_col), alpha=t_in)

    # ---- step 2 (frames 31-56): right "?" -> orange minus that moves up
    t_in2 = smooth((i - 31) / 6.0)
    t_mv = smooth((i - 35) / 21.0)
    if i >= 31:
        fade_box(img, Q_RIGHT, t_in2)
        draw_minus(img, right_cx, cy0 - UP_SHIFT * t_mv, ORANGE, alpha=t_in2)
    return np.asarray(img)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    w = imageio.get_writer(OUT, fps=FPS, codec="libx264", pixelformat="yuv420p",
                           macro_block_size=1, ffmpeg_params=["-crf", "12"])
    for i in range(N):
        w.append_data(frame(i))
    w.close()
    Image.fromarray(frame(N - 1)).save(os.path.join(HERE, "output", "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
