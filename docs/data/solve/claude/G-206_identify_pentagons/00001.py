#!/usr/bin/env python3
"""Find the single pentagon in first_frame.png and mark it with a red circle
that grows from the shape's centre outward until it encircles the shape."""
import os, subprocess, shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 30, 16
RED = (220, 30, 30)
LINE_W = 5
SS = 4  # supersampling factor for anti-aliased ring


def find_pentagon(img):
    """Return (cx, cy, radius) of the pentagon in the RGB image."""
    bg = img[0, 0].astype(int)
    mask = (np.abs(img.astype(int) - bg).sum(axis=2) > 40).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        n = len(approx)
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        pts = c.reshape(-1, 2).astype(float)
        r = np.sqrt(((pts - [cx, cy]) ** 2).sum(1)).max()
        cands.append((n, cx, cy, r))
    pent = [c for c in cands if c[0] == 5]
    assert len(pent) == 1, f"expected exactly one pentagon, got {[c[0] for c in cands]}"
    _, cx, cy, r = pent[0]
    return cx, cy, r


def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep: gentle start and finish, paced over full clip


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W = base.shape[:2]
    cx, cy, r_shape = find_pentagon(base)
    r_final = r_shape + 14  # ring sits just outside the shape

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        frame = base.copy()
        if i > 0:
            r = ease(t) * r_final
            # draw anti-aliased ring on supersampled layer, then composite
            layer = Image.new("L", (W * SS, H * SS), 0)
            d = ImageDraw.Draw(layer)
            d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                      outline=255, width=LINE_W * SS)
            alpha = np.array(layer.resize((W, H), Image.LANCZOS)).astype(float) / 255.0
            a = alpha[..., None]
            frame = (frame * (1 - a) + np.array(RED) * a).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(frame).save(os.path.join(tmp, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        OUT,
    ], check=True)
    shutil.rmtree(tmp)
    print(f"wrote {OUT}: pentagon at ({cx:.1f},{cy:.1f}), ring radius {r_final:.1f}")


if __name__ == "__main__":
    main()
