#!/usr/bin/env python3
"""Mark the longest side of the polygon in first_frame.png with a small red
circle at its midpoint, animated over ~25 frames, and encode to H.264 MP4."""
import os, subprocess
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 25
RED = (255, 0, 0)
RADIUS = 10.0  # final circle radius in px


def extract_polygon(img):
    """Return the polygon's vertices (Nx2 float) from the flat-colour image."""
    mask = ((np.abs(img.astype(int) - 255).sum(axis=2)) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    for eps in (2.0, 2.5, 3.0, 4.0, 5.0):
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) == 8:
            break
    return ap.astype(float)


def longest_edge(pts):
    n = len(pts)
    lengths = [np.hypot(*(pts[(i + 1) % n] - pts[i])) for i in range(n)]
    i = int(np.argmax(lengths))
    return i, lengths, pts[i], pts[(i + 1) % n]


def draw_circle(base, center, radius, color, ss=4):
    """Anti-aliased filled circle via supersampled alpha mask, blended onto base."""
    h, w = base.shape[:2]
    cx, cy = center
    x0, x1 = max(0, int(cx - radius - 2)), min(w, int(cx + radius + 3))
    y0, y1 = max(0, int(cy - radius - 2)), min(h, int(cy + radius + 3))
    if x1 <= x0 or y1 <= y0 or radius <= 0:
        return base
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(float)
    alpha = np.zeros((y1 - y0, x1 - x0), float)
    offs = (np.arange(ss) + 0.5) / ss
    for oy in offs:
        for ox in offs:
            d = np.hypot(xs + ox - cx, ys + oy - cy)
            alpha += (d <= radius)
    alpha /= ss * ss
    out = base.copy()
    patch = out[y0:y1, x0:x1].astype(float)
    col = np.array(color, float)
    patch = patch * (1 - alpha[..., None]) + col * alpha[..., None]
    out[y0:y1, x0:x1] = np.clip(patch + 0.5, 0, 255).astype(np.uint8)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    pts = extract_polygon(base)
    idx, lengths, p, q = longest_edge(pts)
    mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
    for i, L in enumerate(lengths):
        print(f"edge {i}: {pts[i].astype(int).tolist()} -> "
              f"{pts[(i + 1) % len(pts)].astype(int).tolist()}  len={L:.1f}"
              + ("  <-- longest" if i == idx else ""))
    print(f"midpoint of longest edge: ({mid[0]:.1f}, {mid[1]:.1f})")

    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(base.copy())  # first frame identical to the input
            continue
        # Circle grows smoothly from nothing to full size over the remaining frames.
        t = f / (N_FRAMES - 1)
        t = 1 - (1 - t) ** 2  # ease-out
        frames.append(draw_circle(base, mid, RADIUS * t, RED))

    h, w = base.shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
           "-r", str(FPS), "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
