#!/usr/bin/env python3
"""Identify hollow (outlined, unfilled) circles in first_frame.png and animate
a red ring being drawn around each one. Output: /app/output/video.mp4."""
import os, subprocess, math
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80
RED = (0, 0, 255)  # BGR


def find_hollow_points(img):
    """Return list of (cx, cy, r) for connected dark components that are hollow."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 128).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    hollow = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 8 or h < 8:
            continue
        disc_area = math.pi * (w / 2.0) * (h / 2.0)
        fill_ratio = area / disc_area
        if fill_ratio < 0.5:  # outline only -> hollow
            hollow.append((x + w / 2.0, y + h / 2.0, max(w, h) / 2.0))
    hollow.sort(key=lambda c: (c[1], c[0]))  # step order: top-to-bottom, left-to-right
    return hollow


def draw_arc(img, cx, cy, r, frac, thickness):
    """Draw a red arc from the top going clockwise covering `frac` of the ring."""
    if frac <= 0:
        return
    end = 360.0 * min(frac, 1.0)
    if frac >= 1.0:
        cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r)), RED,
                   thickness, lineType=cv2.LINE_AA)
    else:
        cv2.ellipse(img, (int(round(cx)), int(round(cy))), (int(round(r)), int(round(r))),
                    -90, 0, end, RED, thickness, lineType=cv2.LINE_AA)


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def main():
    base = cv2.imread(SRC)
    assert base is not None, SRC
    H, W = base.shape[:2]
    hollow = find_hollow_points(base)
    print(f"Found {len(hollow)} hollow point(s): {hollow}")

    ring_gap, thickness = 14, 6
    hold_start = 12          # identify phase: show the original scene
    hold_end = 12            # final result held
    anim_frames = N_FRAMES - hold_start - hold_end
    per = anim_frames / max(len(hollow), 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    for k in range(N_FRAMES):
        frame = base.copy()
        t = k - hold_start
        for j, (cx, cy, r) in enumerate(hollow):
            local = (t - j * per) / per          # progress of ring j in [0,1]
            frac = ease(local) if local < 1 else 1.0
            if k == N_FRAMES - 1:
                frac = 1.0
            draw_arc(frame, cx, cy, r + ring_gap, frac, thickness)
        cv2.imwrite(os.path.join(tmp, f"{k:04d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "6", "-preset", "slow", "-tune", "stillimage",
        "-vf", f"scale={W}:{H}", OUT], check=True)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
