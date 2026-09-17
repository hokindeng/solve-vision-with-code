#!/usr/bin/env python3
"""LEGO assembly step 4: move the orange 2x4 brick from the callout to the
position marked by the red arrow / dashed outline and snap it into place.

Approach: the brick is cut out of the first frame as a sprite (arrow pixels
that overlap it are inpainted with the correct face colours), the callout is
restored behind it, and the sprite is translated along an arc that follows
the red arrow, hovers over the target, then drops and snaps into place.  The
red arrow / dashed outline (with its white halo) is re-drawn on top of every
frame so no other pixel of the scene changes.
"""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 46

# ---------------------------------------------------------------- scene data
BG = (245, 245, 240)
WHITE = (255, 255, 255)
BOX_GRAY = (180, 180, 180)
DARK = (50, 50, 50)
RED = (255, 0, 0)
TOP, SIDE, FRONT, STUD = (255, 165, 28), (215, 117, 20), (177, 96, 16), (255, 181, 30)

# Callout brick hexagon corners (pixel centres of the 1px outline):
#  L (leftmost), T (top), R (right), R' (right-bottom), B' (bottom), L' (left-bottom)
L, T, R = (30, 401), (203, 301), (289, 351)
Rp, Bp, Lp = (289, 411), (116, 511), (30, 461)
B = (116, 451)  # inner corner (top-face bottom vertex)
HEX = [L, T, R, Rp, Bp, Lp]

# Callout box (1px... actually 2px gray border, white fill)
BOX = (70, 331, 240, 501)      # outer bounds inclusive
BOX_BORDER = 2

# Destination: the brick's leftmost corner must land on (552, 889), which is the
# model's left corner (552, 689) moved 4 studs toward the viewer and 4 studs
# along the model (dashed red outline in the frame).
START = np.array(L, float)
DEST = np.array((552, 889), float)
HOVER = 46.0                   # pixels above destination before the drop


def color_mask(img, c):
    return np.all(img == np.array(c, dtype=img.dtype), axis=2)


def half_plane_mask(pts, shape, tol=0.5):
    """Convex polygon mask including boundary pixels within `tol` px."""
    pts = np.asarray(pts, float)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    inside = np.ones(shape, bool)
    c = pts.mean(0)
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        d = q - p
        n = np.array([d[1], -d[0]]) / np.hypot(*d)
        s = np.sign(np.dot(c - p, n))
        dist = ((xx - p[0]) * n[0] + (yy - p[1]) * n[1]) * s
        inside &= dist >= -tol
    return inside


def seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def build_layers(img):
    h, w, _ = img.shape
    red = color_mask(img, RED)
    white = color_mask(img, WHITE)
    bg = color_mask(img, BG)
    gray = color_mask(img, BOX_GRAY)
    near_red = ndimage.binary_dilation(red, structure=np.ones((5, 5), bool))
    overlay = red | (white & near_red)          # arrow + dashed outline + halo

    # Split the red graphics: the dashed target outline stays in the static
    # layer (the brick covers it when placed); the arrow is drawn on top of
    # everything, as it is over the callout brick in the first frame.
    dl, dt, dr, db = DEST, DEST + (173, -100), DEST + (259, -50), DEST + (86, 50)
    corners = [tuple(p) for p in (dl, dt, dr, db)]
    lower = [(x, y + 60) for x, y in corners]
    box_edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    box_edges += [(lower[i], lower[(i + 1) % 4]) for i in range(4)]
    box_edges += list(zip(corners, lower))
    ys, xs = np.where(red)
    d = np.min([seg_dist(xs, ys, a, b) for a, b in box_edges], axis=0)
    red_outline = np.zeros_like(red)
    red_outline[ys[d <= 2.5], xs[d <= 2.5]] = True
    red_arrow = red & ~red_outline
    arrow = red_arrow | (white & ndimage.binary_dilation(red_arrow, structure=np.ones((5, 5), bool)))

    # --- sprite mask: brick hexagon, minus stray background/box pixels
    hexm = half_plane_mask(HEX, (h, w), 0.5)
    sprite_mask = hexm & ~(bg | gray | (white & ~near_red))

    # --- sprite pixels, with the arrow that overlaps the brick inpainted
    sprite = img.copy()
    inpaint = sprite_mask & overlay
    faces = [
        (half_plane_mask([L, T, R, B], (h, w), 0.5), TOP),
        (half_plane_mask([R, B, Bp, Rp], (h, w), 0.5), SIDE),
        (half_plane_mask([L, B, Bp, Lp], (h, w), 0.5), FRONT),
    ]
    for fm, col in faces:
        sel = inpaint & fm
        sprite[sel] = col
    ys, xs = np.where(inpaint)
    edges = [(L, T), (T, R), (R, Rp), (Rp, Bp), (Bp, Lp), (Lp, L), (L, B), (B, R), (B, Bp)]
    d = np.min([seg_dist(xs, ys, a, b) for a, b in edges], axis=0)
    sprite[ys[d <= 0.5], xs[d <= 0.5]] = DARK

    # --- static background: scene with the brick removed from the callout
    static = img.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0, x1, y1 = BOX
    in_box = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    in_fill = (xx >= x0 + BOX_BORDER) & (xx <= x1 - BOX_BORDER) & \
              (yy >= y0 + BOX_BORDER) & (yy <= y1 - BOX_BORDER)
    static[sprite_mask] = BG
    static[sprite_mask & in_box] = BOX_GRAY
    static[sprite_mask & in_fill] = WHITE

    return static, sprite, sprite_mask, arrow


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def ease_in(t):
    t = np.clip(t, 0, 1)
    return t * t * t


def brick_offset(i):
    """Integer (dx, dy) translation of the sprite for frame i."""
    hover = DEST + np.array([0.0, -HOVER])
    # timeline (frames): 0-1 hold, 2-31 travel along arc, 31-40 drop, 40-45 hold
    t_travel_0, t_travel_1 = 2, 31
    t_drop_0, t_drop_1 = 31, 40
    if i <= t_travel_0:
        pos = START
    elif i < t_travel_1:
        t = smoothstep((i - t_travel_0) / (t_travel_1 - t_travel_0))
        # quadratic Bezier bulging up-right, like the red arrow
        mid = (START + hover) / 2
        ctrl = mid + np.array([60.0, -180.0])
        pos = (1 - t) ** 2 * START + 2 * (1 - t) * t * ctrl + t ** 2 * hover
    elif i < t_drop_1:
        t = ease_in((i - t_drop_0) / (t_drop_1 - t_drop_0))
        pos = hover + (DEST - hover) * t
    else:
        pos = DEST
    off = np.rint(pos - START).astype(int)
    return int(off[0]), int(off[1])


def render_frame(static, sprite, sprite_mask, overlay_px, overlay_mask, dx, dy):
    frame = static.copy()
    ys, xs = np.where(sprite_mask)
    frame[ys + dy, xs + dx] = sprite[ys, xs]
    frame[overlay_mask] = overlay_px
    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = np.array(Image.open(SRC).convert("RGB"))
    static, sprite, sprite_mask, overlay_mask = build_layers(img)
    overlay_px = img[overlay_mask]

    frames = []
    for i in range(N_FRAMES):
        dx, dy = brick_offset(i)
        f = render_frame(static, sprite, sprite_mask, overlay_px, overlay_mask, dx, dy)
        if i == 0:
            assert np.array_equal(f, img), "frame 0 must reproduce first_frame.png"
        frames.append(f)

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for name in os.listdir(tmp):
        os.remove(os.path.join(tmp, name))
    os.rmdir(tmp)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
