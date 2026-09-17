#!/usr/bin/env python3
"""Domino chain-reaction video: START -> T1..T3 -> (A1..A4 up, B1..B2 down)."""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 62

first = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)
BG = first[5, 5].copy()

# Domino rectangles (x0, x1, y0, y1 inclusive), plus a sprite bbox that may be
# wider to include label text that overflows the rectangle (START).
DOMINOS = {
    "START": dict(rect=(74, 115, 441, 541), bbox=(58, 132, 441, 541)),
    "T1": dict(rect=(197, 236, 441, 541)),
    "T2": dict(rect=(319, 358, 441, 541)),
    "T3": dict(rect=(441, 480, 441, 541)),
    "A1": dict(rect=(563, 602, 333, 433)),
    "A2": dict(rect=(685, 724, 315, 415)),
    "A3": dict(rect=(807, 846, 297, 397)),
    "A4": dict(rect=(929, 968, 280, 380)),
    "B1": dict(rect=(563, 602, 549, 649)),
    "B2": dict(rect=(685, 724, 566, 666)),
}

# Chain order and timing (frame at which each domino starts to tip).
STAGES = [["START"], ["T1"], ["T2"], ["T3"], ["A1", "B1"], ["A2", "B2"], ["A3"], ["A4"]]
STAGE_GAP = 6      # frames between successive dominos starting to fall
FALL_FRAMES = 12   # frames for a domino to go from upright to fallen
FIRST_START = 2
FINAL_ANGLE = 84.0  # degrees clockwise (tilted right, resting on the next one)

start_frame = {}
for i, names in enumerate(STAGES):
    for n in names:
        start_frame[n] = FIRST_START + i * STAGE_GAP

# Build sprites (premultiplied RGBA layers, full frame size) and the static background.
background = first.copy()
sprites = {}
for name, d in DOMINOS.items():
    x0, x1, y0, y1 = d.get("bbox", d["rect"])
    region = first[y0:y1 + 1, x0:x1 + 1]
    alpha = (np.abs(region - BG).sum(axis=2) > 12).astype(np.float32)
    layer = np.zeros((H, W, 4), np.float32)
    layer[y0:y1 + 1, x0:x1 + 1, :3] = region * alpha[..., None]
    layer[y0:y1 + 1, x0:x1 + 1, 3] = alpha
    sprites[name] = layer
    background[y0:y1 + 1, x0:x1 + 1] = BG
    rx0, rx1, ry0, ry1 = d["rect"]
    d["pivot"] = (rx1 + 1.0, ry1 + 1.0)  # bottom-right corner of the tile


def angle_at(name, f):
    s = (f - start_frame[name]) / FALL_FRAMES
    s = min(max(s, 0.0), 1.0)
    return FINAL_ANGLE * (s ** 1.8)  # gravity-like ease-in


def render(f):
    out = background.copy()
    # Draw right-to-left so a fallen domino lies on top of the one it knocked over.
    order = ["A4", "B2", "A3", "A2", "B1", "A1", "T3", "T2", "T1", "START"]
    for name in order:
        ang = angle_at(name, f)
        layer = sprites[name]
        if ang > 0:
            M = cv2.getRotationMatrix2D(DOMINOS[name]["pivot"], -ang, 1.0)
            layer = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = layer[..., 3:4]
        out = layer[..., :3] + out * (1 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main():
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        frame = render(f) if f > 0 else first.astype(np.uint8)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
