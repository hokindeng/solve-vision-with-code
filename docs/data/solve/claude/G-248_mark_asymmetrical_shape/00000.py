#!/usr/bin/env python3
"""Find the single asymmetrical shape in first_frame.png and circle it in red,
animating the circle sweep over 16 frames at 16 fps."""
import os, subprocess, shutil
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 16, 16


def asymmetry_score(comp_mask):
    """Min over reflection axes (through centroid) of mismatch fraction.
    Symmetric shapes have some axis with ~0 mismatch."""
    ys, xs = np.nonzero(comp_mask)
    cx, cy = xs.mean(), ys.mean()
    pts = np.stack([xs - cx, ys - cy], 1).astype(np.float64)
    best = 1.0
    h, w = comp_mask.shape
    for deg in np.arange(0, 180, 1.0):
        t = np.deg2rad(deg)
        d = np.array([np.cos(t), np.sin(t)])
        # reflect points across line through origin with direction d
        proj = pts @ d
        refl = 2 * np.outer(proj, d) - pts
        rx = np.rint(refl[:, 0] + cx).astype(int)
        ry = np.rint(refl[:, 1] + cy).astype(int)
        ok = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
        inside = np.zeros(len(pts), bool)
        inside[ok] = comp_mask[ry[ok], rx[ok]] > 0
        best = min(best, 1.0 - inside.mean())
    return best


def main():
    base = cv2.imread(SRC)  # BGR
    bg = np.array([255, 255, 255])
    mask = (np.abs(base.astype(int) - bg).sum(2) > 30).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 200]
    scores = {i: asymmetry_score((lab == i).astype(np.uint8)) for i in comps}
    target = max(scores, key=scores.get)
    for i in comps:
        print(f"component {i}: centroid={cent[i].round(1)}, asymmetry={scores[i]:.4f}")
    print("target:", target)

    x, y, w, h, _ = stats[target]
    cx, cy = x + w / 2.0, y + h / 2.0
    radius = int(round(np.hypot(w, h) / 2.0 + 14))
    thickness = 6
    color = (0, 0, 255)  # red in BGR

    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    start_angle = -90.0
    for f in range(N_FRAMES):
        frame = base.copy()
        if f > 0:
            sweep = 360.0 * f / (N_FRAMES - 1)
            if sweep >= 360.0:
                cv2.circle(frame, (int(round(cx)), int(round(cy))), radius, color,
                           thickness, lineType=cv2.LINE_AA)
            else:
                cv2.ellipse(frame, (int(round(cx)), int(round(cy))), (radius, radius),
                            0, start_angle, start_angle + sweep, color, thickness,
                            lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(tmp, f"{f:03d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
