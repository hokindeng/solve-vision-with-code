#!/usr/bin/env python3
"""Animate the domino chain reaction described in /app/prompt.txt.

Each domino is cut out of first_frame.png as a sprite (box + label) and rotated
clockwise about its bottom-right corner so it ends up tilted to the right.
Everything else in the frame is left untouched.
"""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 50
BG = np.array([240, 240, 240], dtype=np.uint8)
FINAL_ANGLE = 78.0  # degrees clockwise (tilted right)

# name: (box x0, y0, x1, y1 inclusive), (sprite crop x0, y0, x1, y1 inclusive)
# The sprite crop is the box plus any label overflow (START's label spills out).
DOMINOS = {
    "START": ((67, 456, 119, 582), (59, 456, 127, 582)),
    "T1":    ((211, 456, 263, 582), (211, 456, 263, 582)),
    "A1":    ((355, 363, 407, 489), (355, 363, 407, 489)),
    "B1":    ((355, 549, 407, 675), (355, 549, 407, 675)),
    "A2":    ((499, 338, 551, 464), (499, 338, 551, 464)),
    "B2":    ((499, 573, 551, 699), (499, 573, 551, 699)),
    "A3":    ((643, 313, 695, 439), (643, 313, 695, 439)),
    "B3":    ((643, 598, 695, 724), (643, 598, 695, 724)),
}

# Fall schedule: (start frame, duration in frames). Left to right, branches parallel.
FALL_DUR = 10
SCHEDULE = {
    "START": 1,
    "T1": 8,
    "A1": 15, "B1": 15,
    "A2": 22, "B2": 22,
    "A3": 29, "B3": 29,
}


def ease(t):
    """Toppling domino: slow start, accelerating, hard stop."""
    t = min(max(t, 0.0), 1.0)
    return t ** 1.8


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W = base.shape[:2]

    # Static background: original frame with every domino region erased.
    static = base.copy()
    sprites = {}
    for name, (box, crop) in DOMINOS.items():
        cx0, cy0, cx1, cy1 = crop
        patch = base[cy0:cy1 + 1, cx0:cx1 + 1].copy()
        bx0, by0, bx1, by1 = box
        alpha = np.zeros(patch.shape[:2], dtype=np.uint8)
        # Fully opaque inside the box, plus any label pixels that overflow it.
        alpha[by0 - cy0:by1 - cy0 + 1, bx0 - cx0:bx1 - cx0 + 1] = 255
        nonbg = np.any(patch != BG, axis=2)
        alpha[nonbg] = 255
        rgba = np.dstack([patch, alpha])
        sprites[name] = (rgba, (cx0, cy0), (bx1 + 1, by1 + 1))  # pivot = bottom-right corner
        static[cy0:cy1 + 1, cx0:cx1 + 1][alpha > 0] = BG

    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(base.copy())
            continue
        frame = static.copy().astype(np.float32)
        for name in DOMINOS:
            rgba, (ox, oy), (px, py) = sprites[name]
            t = (f - SCHEDULE[name]) / FALL_DUR
            ang = FINAL_ANGLE * ease(t)
            if ang <= 0:
                # Upright: paste the sprite back unchanged.
                h, w = rgba.shape[:2]
                a = rgba[..., 3:4].astype(np.float32) / 255.0
                reg = frame[oy:oy + h, ox:ox + w]
                frame[oy:oy + h, ox:ox + w] = reg * (1 - a) + rgba[..., :3] * a
                continue
            # Place sprite on a full-size transparent canvas and rotate about pivot.
            canvas = np.zeros((H, W, 4), dtype=np.uint8)
            h, w = rgba.shape[:2]
            canvas[oy:oy + h, ox:ox + w] = rgba
            # Clockwise on screen => negative angle for cv2 (which is CCW positive).
            M = cv2.getRotationMatrix2D((float(px), float(py)), -ang, 1.0)
            rot = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            a = rot[..., 3:4].astype(np.float32) / 255.0
            frame = frame * (1 - a) + rot[..., :3].astype(np.float32) * a
        frames.append(np.clip(frame + 0.5, 0, 255).astype(np.uint8))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "8", "-preset", "medium", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames @ {FPS} fps")


if __name__ == "__main__":
    main()
