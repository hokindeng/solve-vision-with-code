#!/usr/bin/env python3
"""LEGO assembly step 4: move the blue 2x4 brick from the callout box to the
position marked by the red arrow / dashed ghost outline and snap it in place.

Everything is derived from /app/first_frame.png:
  * the brick sprite is cut out of the callout (arrow pixels drawn over it are
    reconstructed from the flat face colours and outline), the hole is refilled
    with the callout box's white fill + its hidden border lines, and the sprite
    is composited along a smooth path onto the model, under the arrow overlay.
    The dashed ghost outline fades out as the brick snaps into place.
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 46

BG = np.array([245, 245, 240], np.uint8)

# Brick silhouette in the callout (measured from first_frame.png):
# L, B (top/back), R, R', F' (bottom/front), L'
HEX = np.array([[30, 401], [203, 301], [290, 351],
                [290, 411], [116, 511], [30, 461]], np.int32)
# Callout box rectangle (2 px lines, colour 180) partly hidden by the brick.
BOX = (70, 331, 240, 501)
BOX_COL = np.array([180, 180, 180], np.uint8)
BOX_FILL = np.array([255, 255, 255], np.uint8)
# Flat colours of the brick faces / outline (sampled from the source frame).
TOP_COL = np.array([0, 102, 229], np.uint8)
FRONT_COL = np.array([0, 72, 162], np.uint8)
LEFT_COL = np.array([0, 59, 133], np.uint8)
OUTLINE_COL = np.array([50, 50, 50], np.uint8)
# Final translation: brick's back-top edge meets the model's front-left edge,
# exactly where the dashed ghost outline is drawn.
DEST = np.array([262.0, 444.0])


def load():
    return np.array(Image.open(SRC).convert("RGB"))


def _fit_line_pixels(dark, exclude, p, q, hex_b):
    """Pixels of the 1-px outline segment p-q, reconstructed under `exclude`.

    Fits the segment to the exact outline pixels visible outside `exclude`
    and returns the (x, y) coordinates the line occupies inside `exclude`."""
    (x0, y0), (x1, y1) = p, q
    vertical = abs(x1 - x0) < abs(y1 - y0)
    h, w = dark.shape
    ys, xs = np.nonzero(dark & ~exclude & hex_b)
    # distance of each dark pixel from the ideal segment
    px, py = xs.astype(float), ys.astype(float)
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0, 1)
    d = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    sel = d < 1.3
    xs, ys = xs[sel], ys[sel]
    out = []
    if vertical:
        if len(ys) >= 2:
            m, b = np.polyfit(ys, xs, 1)
        else:
            m, b = 0.0, float(x0)
        for y in range(min(y0, y1), max(y0, y1) + 1):
            x = int(round(m * y + b))
            if 0 <= x < w and exclude[y, x]:
                out.append((x, y))
    else:
        if len(xs) >= 2:
            m, b = np.polyfit(xs, ys, 1)
        else:
            m, b = dy / dx, y0 - dy / dx * x0
        for x in range(min(x0, x1), max(x0, x1) + 1):
            y = int(round(m * x + b))
            if 0 <= y < h and exclude[y, x]:
                out.append((x, y))
    return out


def build_layers(img):
    """Return (base_with_ghost, base_without_ghost, sprite_rgba, sprite_origin,
    arrow_overlay_mask)."""
    h, w = img.shape[:2]
    R = img[..., 0].astype(int)

    hex_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hex_mask, [HEX], 1)
    hex_b = hex_mask.astype(bool)

    # --- the red arrow (with its white halo) drawn over the brick ------------
    arrow_over_brick = hex_b & (R > 100)
    fill = cv2.dilate(arrow_over_brick.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & hex_b

    # --- sprite: brick with the arrow pixels reconstructed --------------------
    L, Bk, Rt, Rb, Fb, Lb = [tuple(int(v) for v in p) for p in HEX]
    F = (Fb[0], Fb[1] - 60)  # front-top corner (brick is 60 px tall)
    faces = [  # (polygon, flat colour) — sampled from the source rendering
        (np.array([L, Bk, Rt, F], np.int32), TOP_COL),
        (np.array([F, Rt, Rb, Fb], np.int32), FRONT_COL),
        (np.array([L, F, Fb, Lb], np.int32), LEFT_COL),
    ]
    sprite_rgb = img.copy()
    for poly, col in faces:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly], 1)
        sprite_rgb[fill & m.astype(bool)] = col
    dark = np.all(img == OUTLINE_COL, axis=2)
    lines = [(L, Bk), (Bk, Rt), (Rt, Rb), (Rb, Fb), (Fb, Lb), (Lb, L),
             (F, L), (F, Rt), (F, Fb)]
    for p, q in lines:
        for x, y in _fit_line_pixels(dark, fill, p, q, hex_b):
            sprite_rgb[y, x] = OUTLINE_COL

    x0, y0 = HEX[:, 0].min(), HEX[:, 1].min()
    x1, y1 = HEX[:, 0].max() + 1, HEX[:, 1].max() + 1
    alpha = (hex_mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    sprite = np.dstack([sprite_rgb[y0:y1, x0:x1], alpha])

    # --- base frame: brick removed from the callout ----------------------------
    base = img.copy()
    base[hex_b] = BG
    bx0, by0, bx1, by1 = BOX
    inner = np.zeros((h, w), bool)
    inner[by0:by1 + 1, bx0:bx1 + 1] = True
    base[inner & hex_b] = BOX_FILL           # callout box has a white fill
    box = np.zeros((h, w), bool)
    box[by0:by0 + 2, bx0:bx1 + 1] = True
    box[by1 - 1:by1 + 1, bx0:bx1 + 1] = True
    box[by0:by1 + 1, bx0:bx0 + 2] = True
    box[by0:by1 + 1, bx1 - 1:bx1 + 1] = True
    base[box & hex_b] = BOX_COL
    # arrow pixels stay exactly as in the source frame
    base[arrow_over_brick] = img[arrow_over_brick]

    # --- arrow overlay (arrow + white halo): drawn above everything ------------
    red = (img[..., 0] > 180) & (img[..., 1] < 80) & (img[..., 2] < 80)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    # arrow shaft and arrowhead are separate components (white halo gap);
    # the dashed ghost outline consists of many tiny components.
    arrow_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 150]
    arrow = np.isin(lab, arrow_ids)
    near = cv2.dilate(arrow.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    whiteish = np.all(img >= 240, axis=2) & ~np.all(img == BG, axis=2)
    overlay = arrow | (near & whiteish)

    # --- base without the dashed ghost outline (revealed when the brick lands) --
    ghost = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if i not in arrow_ids and stats[i, cv2.CC_STAT_TOP] > 600:
            ghost[lab == i] = 1
    ghost = cv2.dilate(ghost, np.ones((3, 3), np.uint8))
    base_clean = cv2.inpaint(base, ghost, 3, cv2.INPAINT_TELEA)

    return base, base_clean, sprite, (x0, y0), overlay


def composite(base, sprite, origin, offset):
    """Alpha-composite sprite translated by a (possibly fractional) offset."""
    out = base.copy()
    sh, sw = sprite.shape[:2]
    ox, oy = origin
    tx, ty = float(offset[0]), float(offset[1])
    # pad sprite by 2 px so bilinear resampling has room
    pad = 2
    canvas = np.zeros((sh + 2 * pad, sw + 2 * pad, 4), np.uint8)
    canvas[pad:pad + sh, pad:pad + sw] = sprite
    ix, iy = int(np.floor(tx)), int(np.floor(ty))
    fx, fy = tx - ix, ty - iy
    M = np.float32([[1, 0, fx], [0, 1, fy]])
    shifted = cv2.warpAffine(canvas, M, (sw + 2 * pad, sh + 2 * pad),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    px, py = ox + ix - pad, oy + iy - pad
    ph, pw = shifted.shape[:2]
    # clip to frame
    sx0, sy0 = max(0, -px), max(0, -py)
    dx0, dy0 = max(0, px), max(0, py)
    dx1, dy1 = min(W, px + pw), min(H, py + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return out
    src = shifted[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    a = src[..., 3:4].astype(np.float32) / 255.0
    dst = out[dy0:dy1, dx0:dx1].astype(np.float32)
    out[dy0:dy1, dx0:dx1] = np.clip(src[..., :3] * a + dst * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    return out


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def bezier(p0, p1, p2, p3, t):
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def brick_offset(i):
    """Screen-space offset of the brick for frame i, plus ghost visibility."""
    HOLD0, TRAVEL_END, LAND = 2, 34, 41
    HOVER = 50.0  # px above the target before the final drop
    hover = DEST + np.array([0.0, -HOVER])
    if i <= HOLD0:
        return np.zeros(2), 1.0
    if i <= TRAVEL_END:
        t = smoothstep((i - HOLD0) / (TRAVEL_END - HOLD0))
        p0 = np.zeros(2)
        p1 = np.array([70.0, -90.0])          # lift out of the callout
        p2 = np.array([hover[0], hover[1] - 160.0])  # come in from above
        return bezier(p0, p1, p2, hover, t), 1.0
    if i <= LAND:
        t = (i - TRAVEL_END) / (LAND - TRAVEL_END)
        t = t * t                            # accelerate into the snap
        pos = hover + (DEST - hover) * t
        return pos, 1.0 - smoothstep((t - 0.4) / 0.6)
    return DEST.copy(), 0.0


def render_frames():
    img = load()
    base, base_clean, sprite, origin, overlay = build_layers(img)
    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(img.copy())  # first frame identical to the source
            continue
        off, ghost_vis = brick_offset(i)
        if ghost_vis >= 1.0:
            bg = base
        elif ghost_vis <= 0.0:
            bg = base_clean
        else:
            bg = np.clip(base.astype(np.float32) * ghost_vis
                         + base_clean.astype(np.float32) * (1 - ghost_vis) + 0.5,
                         0, 255).astype(np.uint8)
        if np.allclose(off, DEST):
            off = np.round(off)  # crisp integer placement once snapped
        fr = composite(bg, sprite, origin, off)
        fr[overlay] = img[overlay]  # the red arrow is an annotation layer on top
        frames.append(fr)
    return frames


def write_video(frames):
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


if __name__ == "__main__":
    frames = render_frames()
    write_video(frames)
    # also keep a few stills for inspection
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")
