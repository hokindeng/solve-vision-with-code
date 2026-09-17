#!/usr/bin/env python3
"""Generate the domino chain-reaction video from first_frame.png.

Every domino tile (rectangle + label) is cut straight out of the first frame,
rotated clockwise about its bottom-right corner, and composited back over the
background with the original domino erased.  Nothing else in the frame changes.
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 54
BG = 252
FINAL_ANGLE = 78.0     # degrees clockwise ("tilted right", resting on the next domino)
FALL_FRAMES = 12       # frames for one domino to fall

# (name, tile bbox x0,x1,y0,y1 inclusive, pivot (bottom-right corner of body), stage)
# START's bbox is widened to include the label overflow.
DOMINOS = [
    ("START", (53, 121, 447, 569), (114, 569), 0),
    ("T1",    (201, 255, 447, 569), (255, 569), 1),
    ("T2",    (342, 396, 447, 569), (396, 569), 2),
    ("A1",    (483, 537, 356, 478), (537, 478), 3),
    ("B1",    (483, 537, 538, 660), (537, 660), 3),
    ("A2",    (624, 678, 337, 459), (678, 459), 4),
    ("B2",    (624, 678, 556, 678), (678, 678), 4),
    ("A3",    (765, 819, 319, 441), (819, 441), 5),
]
STAGE_START = [2, 9, 16, 23, 30, 37]   # frame each stage begins falling


def ease(t):
    """Gravity-like fall: slow start, fast finish, tiny settle at the end."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t) if t < 1 else 1.0


def angle_at(frame, stage):
    start = STAGE_START[stage]
    if frame <= start:
        return 0.0
    t = (frame - start) / FALL_FRAMES
    # accelerate (t^2 shape) then settle
    a = FINAL_ANGLE * min(1.0, t) ** 1.8
    return a


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W = base.shape[:2]

    # Extract RGBA layers (full canvas) for each domino and erase them from the background.
    background = base.copy()
    layers = []
    for name, (x0, x1, y0, y1), pivot, stage in DOMINOS:
        layer = np.zeros((H, W, 4), np.uint8)
        tile = base[y0:y1 + 1, x0:x1 + 1]
        layer[y0:y1 + 1, x0:x1 + 1, :3] = tile
        # tile alpha: opaque where not background colour
        alpha = (np.abs(tile.astype(int) - BG).sum(2) > 0).astype(np.uint8) * 255
        # the body rectangle itself is fully opaque (fill may equal bg nowhere, but be safe)
        layer[y0:y1 + 1, x0:x1 + 1, 3] = alpha
        layers.append((layer, pivot, stage))
        background[y0:y1 + 1, x0:x1 + 1] = BG

    os.makedirs(OUT_DIR, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)

    for f in range(N_FRAMES):
        if f == 0:
            frame = base.copy()          # exact first frame
        else:
            frame = background.astype(np.float32)
            # Composite right-to-left so an earlier (fallen) domino rests on top of the next.
            for layer, (px, py), stage in reversed(layers):
                ang = angle_at(f, stage)
                if ang == 0.0:
                    rot = layer
                else:
                    # cv2 positive angle = counter-clockwise; we fall clockwise (to the right)
                    M = cv2.getRotationMatrix2D((float(px), float(py)), -ang, 1.0)
                    rot = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                a = rot[:, :, 3:4].astype(np.float32) / 255.0
                frame = frame * (1 - a) + rot[:, :, :3].astype(np.float32) * a
            frame = np.clip(frame + 0.5, 0, 255).astype(np.uint8)
        ffmpeg.stdin.write(frame.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
