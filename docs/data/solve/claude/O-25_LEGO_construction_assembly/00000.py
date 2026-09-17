#!/usr/bin/env python3
"""Generate LEGO assembly step-3 video: move the red 2x2 brick from the callout
box to the position indicated by the red arrow / dashed outline and snap it in."""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 46, 16

# --- measured scene constants -------------------------------------------------
BG = (245, 245, 240)
WHITE = (255, 255, 255)
ARROW_RED = (255, 0, 0)
OUTLINE = (50, 50, 50)
TOP, LEFT, RIGHT, STUD = (241, 31, 10), (140, 18, 6), (170, 22, 7), (255, 34, 11)
A, B, H = 87, 50, 60          # half-width, half-height of top diamond, brick height
U = (A / 2.0, -B / 2.0)       # one stud along the "back-right" axis
V = (A / 2.0, B / 2.0)        # one stud along the "front-right" axis
SA = 43                       # stud spacing (x) used by the source renderer
L0 = (73, 376)                # callout brick: left vertex of top face
L1 = (508, 872)               # destination: left vertex of top face
CALLOUT_FRAME = (180, 180, 180)  # 2-px frame of the callout box (x 70-240, y 331-501)


def brick_polys(L):
    lx, ly = L
    T = (lx + A, ly - B)
    R = (lx + 2 * A, ly)
    F = (lx + A, ly + B)
    dn = lambda p: (p[0], p[1] + H)
    top = [L, T, R, F]
    left = [L, F, dn(F), dn(L)]
    right = [F, R, dn(R), dn(F)]
    hexa = [L, T, R, dn(R), dn(F), dn(L)]
    return top, left, right, hexa


def draw_brick(img, L):
    """Draw the 2x2 red brick with its top-face left vertex at integer L."""
    d = ImageDraw.Draw(img)
    top, left, right, _ = brick_polys(L)
    d.polygon(left, fill=LEFT, outline=OUTLINE)
    # the source renderer shows the hidden back-bottom edge across the left face
    d.line([(L[0], L[1] + H), (L[0] + A, L[1] - B + H)], fill=OUTLINE)
    d.polygon(right, fill=RIGHT, outline=OUTLINE)
    d.polygon(top, fill=TOP, outline=OUTLINE)
    for i in range(2):
        for j in range(2):
            # integer stud boxes matching the source rasterisation exactly
            x0 = L[0] + 25 + SA * (i + j); x1 = x0 + 37
            cy = L[1] + 25 * (j - i)
            y0 = cy - 10; y1 = cy + 10
            d.ellipse([x0, y0, x1, y1], fill=STUD, outline=OUTLINE)


def polygon_mask(poly, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(poly, fill=255, outline=255)
    return np.array(m) > 0


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def build_scene_without_brick(base):
    """Remove the brick from the callout box (keep the arrow drawn over it).
    The brick slightly overhangs the callout frame, so restore the frame colour
    and the page background where appropriate."""
    img = base.copy()
    _, _, _, hexa = brick_polys(L0)
    m = dilate(polygon_mask(hexa, img.shape[:2]), 1)
    palette = np.zeros(m.shape, bool)
    for c in (TOP, LEFT, RIGHT, STUD, OUTLINE):
        palette |= (img == np.array(c)).all(-1)
    fill = m & palette
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    inner = (xx >= 72) & (xx <= 238) & (yy >= 333) & (yy <= 499)
    frame = (xx >= 70) & (xx <= 240) & (yy >= 331) & (yy <= 501)
    img[fill & inner] = WHITE
    img[fill & frame & ~inner] = CALLOUT_FRAME
    img[fill & ~frame] = BG
    return img


def erase_dashed_target(img, base):
    """Inpaint the red dashed target outline where it is not covered by the
    placed brick (a 1-px sliver around the hexagon and the part below it),
    using the nearest non-red pixel in the same row."""
    red = (base == np.array(ARROW_RED)).all(-1)
    _, _, _, hexa = brick_polys(L1)
    poly = polygon_mask(hexa, img.shape[:2])
    region = dilate(poly, 2) & ~poly
    region[845:1000, 495:700] = True
    ys, xs = np.nonzero(red & region)
    for y, x in zip(ys, xs):
        for k in range(1, 12):
            for xx in (x - k, x + k):
                if not red[y, xx]:
                    img[y, x] = base[y, xx]; break
            else:
                continue
            break
    return img


def arrow_path(base):
    """Centerline of the red arrow (from callout brick to arrowhead), resampled
    by arc length. Returns callable s in [0,1] -> (x, y)."""
    red = (base == np.array(ARROW_RED)).all(-1)
    pts = []
    for x in range(L0[0] + A, L1[0] + A + 1):
        col = np.nonzero(red[:845, x])[0]
        if len(col):
            pts.append((x, col.mean()))
    pts = np.array(pts, float)
    # light smoothing of the centerline
    k = 9
    ker = np.ones(k) / k
    ysm = np.convolve(np.pad(pts[:, 1], (k // 2, k // 2), mode="edge"), ker, mode="valid")
    pts[:, 1] = ysm
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0], np.cumsum(seg)]); cum /= cum[-1]

    def at(s):
        s = float(np.clip(s, 0, 1))
        x = np.interp(s, cum, pts[:, 0]); y = np.interp(s, cum, pts[:, 1])
        return x, y
    return at, pts[0], pts[-1]


def smoothstep(t):
    t = float(np.clip(t, 0, 1)); return t * t * (3 - 2 * t)


def brick_position(f, path, p_start, p_end):
    """Return (L, snapped) for frame index f."""
    F_FLY_END, F_HOVER_END, F_SNAP = 30, 32, 40
    hover = (L1[0], L1[1] - H)          # aligned right above the destination
    if f <= F_FLY_END:
        s = smoothstep(f / F_FLY_END)
        ax, ay = path(s)
        # offset of the brick's left vertex from the arrow centreline, blended
        off0 = (L0[0] - p_start[0], L0[1] - p_start[1])
        off1 = (hover[0] - p_end[0], hover[1] - p_end[1])
        x = ax + off0[0] + (off1[0] - off0[0]) * s
        y = ay + off0[1] + (off1[1] - off0[1]) * s
        return (int(round(x)), int(round(y))), False
    if f <= F_HOVER_END:
        return hover, False
    if f < F_SNAP:
        t = (f - F_HOVER_END) / (F_SNAP - F_HOVER_END)
        t = t * t                        # accelerate downwards, then snap
        return (L1[0], int(round(hover[1] + H * t))), False
    return L1, True


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    scene = build_scene_without_brick(base)
    scene_done = erase_dashed_target(scene.copy(), base)
    path, p_start, p_end = arrow_path(base)

    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(base.copy()); continue
        L, snapped = brick_position(f, path, p_start, p_end)
        img = Image.fromarray(scene_done if snapped else scene)
        draw_brick(img, L)
        frames.append(np.array(img))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "12", "-preset", "slow", "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
