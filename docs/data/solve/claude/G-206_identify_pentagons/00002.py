#!/usr/bin/env python3
"""Identify the single pentagon in first_frame.png and mark it with a red
circle that expands from the inside out until it encircles the shape."""
import os
import subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 30
FPS = 16
RED = (220, 30, 30)  # RGB
THICKNESS = 5
MARGIN = 14


def find_pentagon(img_rgb):
    bg = img_rgb[0, 0].astype(int)
    mask = (np.abs(img_rgb.astype(int) - bg).sum(2) > 30).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(ap) == 5:
            (x, y), r = cv2.minEnclosingCircle(c)
            best = (x, y, r)
    if best is None:
        raise RuntimeError("no pentagon found")
    return best


def ease(t):
    return 1 - (1 - t) ** 3  # ease-out cubic


def draw_ring(base_rgb, cx, cy, r, thickness, color, ss=4):
    """Anti-aliased ring composited onto base via supersampled coverage mask."""
    h, w = base_rgb.shape[:2]
    if r <= 0:
        return base_rgb.copy()
    pad = int(r + thickness + 4)
    x0, x1 = max(0, int(cx - pad)), min(w, int(cx + pad) + 1)
    y0, y1 = max(0, int(cy - pad)), min(h, int(cy + pad) + 1)
    big = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    cv2.circle(big, (int(round((cx - x0) * ss)), int(round((cy - y0) * ss))),
               int(round(r * ss)), 255, int(round(thickness * ss)), cv2.LINE_AA)
    cov = big.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3)) / 255.0
    out = base_rgb.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    col = np.array(color, np.float32)
    a = cov[..., None]
    out[y0:y1, x0:x1] = (region * (1 - a) + col * a + 0.5).astype(np.uint8)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = cv2.cvtColor(cv2.imread(FIRST), cv2.COLOR_BGR2RGB)
    cx, cy, r_shape = find_pentagon(base)
    r_final = r_shape + MARGIN

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i in range(N_FRAMES):
        if i == 0:
            frame = base.copy()
        else:
            t = i / (N_FRAMES - 1)
            r = r_final * ease(t)
            frame = draw_ring(base, cx, cy, r, THICKNESS, RED)
        cv2.imwrite(os.path.join(tmp, f"{i:04d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-r", str(FPS), OUT,
    ], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
