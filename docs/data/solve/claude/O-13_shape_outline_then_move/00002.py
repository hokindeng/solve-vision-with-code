"""Animate the bottom-row circle: fill -> outline (step 1), then move to the canvas center (step 2)."""
import os
import numpy as np
from PIL import Image, ImageDraw
import imageio.v3 as iio

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")

W = H = 1024
FPS = 16
N = 64

ORANGE = np.array([229, 120, 11], dtype=np.float64)
WHITE = np.array([255, 255, 255], dtype=np.float64)

# Circle measured from first_frame.png (hard-edged PIL-style rendering).
CX0, CY0, R = 188.0, 682.0, 80.0
CX1, CY1 = 512.0, 512.0          # "move center" -> centre of the canvas
STROKE = 3.0                      # outline width used on the top row


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def bbox(cx, cy):
    cx, cy, r = int(round(cx)), int(round(cy)), int(R)
    return [cx - r, cy - r, cx + r, cy + r]


def disc_mask(cx, cy):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse(bbox(cx, cy), fill=255)
    return np.array(m) > 0


def render(base, cx, cy, fill_alpha):
    """fill_alpha: 1 = fully filled, 0 = outline only (3 px stroke, same bounding box)."""
    img = Image.fromarray(base.copy())
    d = ImageDraw.Draw(img)
    inner = tuple(int(round(v)) for v in (WHITE + (ORANGE - WHITE) * fill_alpha))
    bb = bbox(cx, cy)
    d.ellipse(bb, fill=inner)                                   # interior (fades orange -> white)
    d.ellipse(bb, outline=tuple(int(v) for v in ORANGE), width=int(STROKE))  # outline stays
    return np.array(img)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    # Base frame: everything except the animated circle (its region is plain white background).
    base = first.copy()
    base[disc_mask(CX0, CY0)] = 255
    # Every non-background pixel outside the original circle is a static element; it is
    # re-applied on top of each frame so nothing else ever changes.
    static = np.any(base != 255, axis=2)

    # Timeline (frames): 0-27 style change, 28-33 hold, 34-59 move, 60-63 hold.
    S0, S1 = 0, 27
    M0, M1 = 34, 59

    frames = [first]  # exact first frame
    for i in range(1, N):
        if i <= S1:
            a = 1.0 - ease((i - S0) / (S1 - S0))
            cx, cy = CX0, CY0
        elif i < M0:
            a = 0.0
            cx, cy = CX0, CY0
        elif i <= M1:
            a = 0.0
            t = ease((i - M0) / (M1 - M0))
            cx = CX0 + (CX1 - CX0) * t
            cy = CY0 + (CY1 - CY0) * t
        else:
            a = 0.0
            cx, cy = CX1, CY1
        fr = render(base, cx, cy, a)
        fr[static] = first[static]
        frames.append(fr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    iio.imwrite(
        OUT,
        np.stack(frames),
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", "1", "-preset", "slow", "-tune", "stillimage"],
    )
    # Also save key frames for inspection.
    Image.fromarray(frames[-1]).save(os.path.join(APP, "output", "last_frame.png"))
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
