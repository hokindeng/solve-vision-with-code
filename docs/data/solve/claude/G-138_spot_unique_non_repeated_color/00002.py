#!/usr/bin/env python3
"""Find the shape with a unique color and progressively outline it in black."""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 21
THICKNESS = 6


def find_unique_shape(img):
    """Return the binary mask of the single shape whose color occurs only once."""
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    # background = most frequent color
    bg = colors[np.argmax(counts)]
    candidates = []
    for col, cnt in zip(colors, counts):
        if np.array_equal(col, bg) or cnt < 500:
            continue  # ignore background and anti-alias specks
        mask = np.all(img == col, axis=2).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 500]
        candidates.append((col, len(comps), labels, comps))
    uniques = [c for c in candidates if c[1] == 1]
    assert len(uniques) == 1, f"expected exactly one unique color, got {len(uniques)}"
    col, _, labels, comps = uniques[0]
    return (labels == comps[0]).astype(np.uint8), col


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    mask, col = find_unique_shape(base)
    print("unique color:", col.tolist(), "area:", int(mask.sum()))

    # Outer contour of the shape, ordered around the boundary.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    # Start tracing from the topmost point for a natural clockwise stroke.
    start = int(np.argmin(contour[:, 1]))
    contour = np.roll(contour, -start, axis=0)
    # OpenCV orders outer contours counter-clockwise (in image coords); reverse for clockwise.
    contour = np.concatenate([contour[:1], contour[1:][::-1]], axis=0)
    total = len(contour)

    frames = []
    for f in range(N_FRAMES):
        frame = base.copy()
        if f == 0:
            frames.append(frame)
            continue
        # Progress over frames 1..N-1, with smooth ease-in-out pacing.
        t = f / (N_FRAMES - 1)
        t = 0.5 - 0.5 * np.cos(np.pi * t)
        k = int(round(total * t))
        if f == N_FRAMES - 1:
            k = total
        if k >= 2:
            pts = contour[:k].reshape(-1, 1, 2).astype(np.int32)
            closed = k >= total
            cv2.polylines(frame, [pts], isClosed=closed, color=(0, 0, 0),
                          thickness=THICKNESS, lineType=cv2.LINE_8)
        frames.append(frame)

    # Encode with ffmpeg: H.264, yuv420p, 16 fps.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
