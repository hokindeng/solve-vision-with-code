#!/usr/bin/env python3
"""LEGO assembly step 4: move the blue 2x4 brick from the callout onto the model.

The source frame is rendered with flat, non-antialiased colours, so the brick is cut out
pixel-exactly, the callout box behind it is reconstructed, and the brick is translated
(integer offsets, nearest neighbour) along a lift -> arc -> hover -> snap-down path.
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
N_FRAMES = 46
FPS = 16

BG = np.array([245, 245, 240], np.uint8)
WHITE = np.array([255, 255, 255], np.uint8)
GRAY = np.array([180, 180, 180], np.uint8)
RED = np.array([255, 0, 0], np.uint8)
DARK = np.array([50, 50, 50], np.uint8)
TOP = np.array([0, 102, 229], np.uint8)
SIDE_R = np.array([0, 72, 162], np.uint8)

# Translation from the callout brick to the dashed target (measured from the frame).
DX, DY = 478, 260

# Callout box: 2 px gray border on x in {70,71,239,240}, y in {331,332,500,501}; white inside.
BOX_X0, BOX_X1, BOX_Y0, BOX_Y1 = 70, 240, 331, 501

# Brick top-face front-right edge line (top face / right face boundary), hidden by the arrow.
EDGE_P0 = (118.0, 450.0)
EDGE_SLOPE = (352.0 - 450.0) / (289.0 - 118.0)


def load():
    return np.array(Image.open(SRC).convert("RGB"))


def eq(img, col):
    return np.all(img == np.asarray(col, np.uint8), axis=2)


def brick_mask(src):
    """Binary silhouette of the brick (including its 1 px dark outline) and its hull."""
    r, g, b = [src[..., i].astype(int) for i in range(3)]
    blue = (b > 120) & (b > r + 60) & (b > g + 20)
    dark = eq(src, DARK)
    near_blue = cv2.dilate(blue.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    px = blue | (dark & near_blue)
    roi = np.zeros_like(px)
    roi[296:516, 25:297] = True
    px &= roi
    pts = np.column_stack(np.nonzero(px))[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    solid = np.zeros(src.shape[:2], np.uint8)
    cv2.fillPoly(solid, [hull], 255)
    solid[px] = 255  # make sure every brick/outline pixel is included
    outline = np.zeros_like(solid)
    cv2.polylines(outline, [hull], True, 255, thickness=1)
    return solid > 0, outline > 0, hull


def make_sprite(src, solid, outline):
    """RGBA (binary alpha) sprite of the brick with the overlapping arrow removed."""
    rgb = src.copy()
    arrow = solid & (eq(src, RED) | eq(src, WHITE))
    ys, xs = np.nonzero(arrow)
    y_line = EDGE_P0[1] + EDGE_SLOPE * (xs - EDGE_P0[0])
    fill = np.where((ys < y_line)[:, None], TOP, SIDE_R)
    fill[np.abs(ys - y_line) < 0.5] = DARK
    fill[outline[ys, xs]] = DARK
    rgb[ys, xs] = fill
    sprite = np.zeros(src.shape[:2] + (4,), np.uint8)
    sprite[..., :3] = rgb
    sprite[..., 3] = solid.astype(np.uint8) * 255
    return sprite


def make_plate(src, solid):
    """Scene with the brick removed: callout box reconstructed, arrow kept."""
    plate = src.copy()
    ys, xs = np.nonzero(solid)
    fill = np.tile(BG, (len(ys), 1))
    inside = (xs >= BOX_X0) & (xs <= BOX_X1) & (ys >= BOX_Y0) & (ys <= BOX_Y1)
    fill[inside] = GRAY
    interior = (xs >= BOX_X0 + 2) & (xs <= BOX_X1 - 2) & (ys >= BOX_Y0 + 2) & (ys <= BOX_Y1 - 2)
    fill[interior] = WHITE
    keep = eq(src, RED)[ys, xs] | eq(src, WHITE)[ys, xs]  # arrow drawn over the brick
    fill[keep] = src[ys, xs][keep]
    plate[ys, xs] = fill
    return plate


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def offset_at(t):
    """Brick translation (dx, dy) for frame index t (floats; rounded when drawing)."""
    lift = 40.0
    hover = 70.0
    t_lift, t_travel, t_hold, t_drop = 6, 32, 36, 43
    if t <= t_lift:
        return 0.0, -lift * ease_out(t / t_lift)
    if t <= t_travel:
        u = smoothstep((t - t_lift) / (t_travel - t_lift))
        p0 = np.array([0.0, -lift])
        p2 = np.array([DX, DY - hover])
        p1 = (p0 + p2) / 2 + np.array([0.0, -160.0])
        p = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2
        return float(p[0]), float(p[1])
    if t <= t_hold:
        return float(DX), float(DY - hover)
    if t <= t_drop:
        v = (t - t_hold) / (t_drop - t_hold)
        return float(DX), float(DY - hover + hover * v ** 2)
    return float(DX), float(DY)


def composite(plate, sprite, dx, dy):
    """Paste the sprite translated by integer (dx, dy). Returns frame and moved alpha."""
    dx, dy = int(round(dx)), int(round(dy))
    h, w = plate.shape[:2]
    frame = plate.copy()
    moved_alpha = np.zeros((h, w), bool)
    sx0, sy0 = max(0, -dx), max(0, -dy)
    sx1, sy1 = min(w, w - dx), min(h, h - dy)
    if sx1 <= sx0 or sy1 <= sy0:
        return frame, moved_alpha
    sub = sprite[sy0:sy1, sx0:sx1]
    a = sub[..., 3] > 0
    dst = frame[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx]
    dst[a] = sub[..., :3][a]
    moved_alpha[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = a
    return frame, moved_alpha


def clean_dash_fringe(frame, plate, alpha):
    """Remove slivers of the dashed target outline peeking out around the seated brick."""
    core = alpha.astype(np.uint8)
    ring = (cv2.dilate(core, np.ones((5, 5), np.uint8)) > 0) & ~alpha
    bgish = eq(plate, BG).astype(np.uint8)
    near_bg = cv2.dilate(bgish, np.ones((5, 5), np.uint8)) > 0
    kill = ring & eq(plate, RED) & near_bg
    frame = frame.copy()
    frame[kill] = BG
    return frame


def render_frames():
    src = load()
    solid, outline, _ = brick_mask(src)
    sprite = make_sprite(src, solid, outline)
    plate = make_plate(src, solid)
    frames = []
    for t in range(N_FRAMES):
        if t == 0:
            frames.append(src.copy())
            continue
        dx, dy = offset_at(t)
        frame, a = composite(plate, sprite, dx, dy)
        if (int(round(dx)), int(round(dy))) == (DX, DY):
            frame = clean_dash_fringe(frame, plate, a)
        frames.append(frame)
    return frames


def main():
    frames = render_frames()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="lego_frames_")
    for t, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"{t:03d}.png"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(tmp, "%03d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "8", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    subprocess.run(cmd, check=True)
    print("wrote", OUT, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
