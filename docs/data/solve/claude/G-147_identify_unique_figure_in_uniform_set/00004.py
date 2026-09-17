#!/usr/bin/env python3
"""Identify the unique shape in first_frame.png and animate circling it in red."""
import os, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 60
W = H = 1024


def find_unique_shape(img):
    """Return (cx, cy, half_extent) of the shape whose fill ratio differs from the rest."""
    bg = img[0, 0].astype(int)
    mask = (np.abs(img.astype(int) - bg).sum(2) > 30).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 50:
            continue
        comps.append(dict(cx=cent[i][0], cy=cent[i][1], w=w, h=h,
                          fill=a / float(w * h), area=a))
    # Feature vector: fill ratio (shape type) and size; the odd one is farthest from the median.
    feats = np.array([[c["fill"] * 10, c["w"] / 100.0, c["h"] / 100.0] for c in comps])
    med = np.median(feats, axis=0)
    idx = int(np.argmax(np.linalg.norm(feats - med, axis=1)))
    c = comps[idx]
    return c["cx"], c["cy"], max(c["w"], c["h"]) / 2.0


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = cv2.imread(SRC)  # BGR
    assert base.shape[:2] == (H, W)
    cx, cy, half = find_unique_shape(base)
    radius = int(round(half * 1.45 + 8))
    color = (0, 0, 255)  # red in BGR
    thickness = 7

    # Timeline: hold (identify) -> draw arc progressively -> hold on completed result.
    hold_start, draw_end = 12, 52

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)

    S = 4  # supersampling for a smooth anti-aliased arc
    center_s = (int(round(cx * S)), int(round(cy * S)))
    for f in range(N_FRAMES):
        frame = base.copy()
        if f >= hold_start:
            t = min(1.0, (f - hold_start) / float(draw_end - hold_start))
            sweep = 360.0 * ease(t)
            if sweep > 0:
                overlay = np.zeros((H * S, W * S), np.uint8)
                cv2.ellipse(overlay, center_s, (radius * S, radius * S), -90, 0, sweep,
                            255, thickness * S, lineType=cv2.LINE_AA)
                alpha = cv2.resize(overlay, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                a3 = alpha[..., None]
                frame = (frame.astype(np.float32) * (1 - a3) + np.array(color, np.float32) * a3)
                frame = np.clip(frame + 0.5, 0, 255).astype(np.uint8)
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"unique shape at ({cx:.1f},{cy:.1f}); wrote {OUT}")


if __name__ == "__main__":
    main()
