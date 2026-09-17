"""Identify the vertex with the largest interior angle and circle it in red."""
import subprocess, os
import numpy as np
import cv2
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 22, 16
RED = (220, 30, 30)


def find_vertices(img):
    mask = (img.sum(axis=2) < 3 * 128).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    eps = 0.02 * cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(float)
    while len(poly) > 3:  # force to a triangle if approximation left extra points
        eps *= 1.5
        poly = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(float)
    return poly


def interior_angles(pts):
    angs = []
    for i in range(3):
        p, a, b = pts[i], pts[(i + 1) % 3], pts[(i + 2) % 3]
        u, v = a - p, b - p
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        angs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return angs


def draw_arc(img, center, radius, sweep_deg, thickness=4):
    """Draw a red arc from -90 deg clockwise for `sweep_deg` degrees (anti-aliased)."""
    if sweep_deg <= 0:
        return img
    out = img.copy()
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.ellipse(out, (cx, cy), (radius, radius), -90, 0, min(sweep_deg, 360),
                RED, thickness, lineType=cv2.LINE_AA)
    return out


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    pts = find_vertices(base)
    angs = interior_angles(pts)
    idx = int(np.argmax(angs))
    target = pts[idx]
    print("vertices:", pts.tolist())
    print("angles:", [round(a, 1) for a in angs], "-> largest at", target.tolist())

    radius = 40
    frames = [base.copy()]  # frame 0 identical to first_frame
    hold = 4  # hold completed circle at the end
    draw_frames = N_FRAMES - 1 - hold
    for k in range(1, N_FRAMES):
        t = min(k / draw_frames, 1.0)
        t = t * t * (3 - 2 * t)  # ease in/out
        frames.append(draw_arc(base, target, radius, 360 * t))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%03d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
