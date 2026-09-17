#!/usr/bin/env python3
"""Connect the numbered dots in first_frame.png with red lines, one at a time."""
import subprocess
import numpy as np
import cv2

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 70
RED = (0, 0, 255)  # BGR
THICKNESS = 4


def find_dots(img):
    """Return dict label->center, detected from non-white blobs; labels by colour/position."""
    mask = (img.min(axis=2) < 240).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    dots = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a > 500:
            dots.append(((x + w / 2.0, y + h / 2.0), max(w, h) / 2.0, i))
    return dots, lab


def main():
    base = cv2.imread(SRC)
    h, w = base.shape[:2]
    dots, lab = find_dots(base)
    # Map numbers to detected blobs (verified against the image).
    known = {1: (557.5, 252.5), 2: (147.5, 870.5), 3: (595.5, 687.5),
             4: (357.5, 402.5), 5: (768.5, 281.5)}
    centers = {}
    for k, (kx, ky) in known.items():
        (cx, cy), r, _ = min(dots, key=lambda d: (d[0][0] - kx) ** 2 + (d[0][1] - ky) ** 2)
        centers[k] = (cx, cy)
    order = [1, 2, 3, 4, 5]
    segs = [(centers[order[i]], centers[order[i + 1]]) for i in range(len(order) - 1)]

    # Everything that is not background is protected (dots + numbers stay untouched).
    protect = np.zeros((h, w), np.uint8)
    for (cx, cy), r, _ in dots:
        cv2.circle(protect, (int(round(cx)), int(round(cy))), int(np.ceil(r)) + 1, 1, -1)
    protect = protect > 0

    # Timing: frame 0 untouched; each segment grows over `grow` frames then pauses.
    hold_end = 4
    active = N_FRAMES - 1 - hold_end            # 65 frames for 4 segments
    per_seg = active / len(segs)                # ~16.25
    pause = 2

    def progress(frame):
        """Return list of completed fraction per segment at given frame."""
        t = frame - 1
        fr = []
        for s in range(len(segs)):
            start = s * per_seg
            grow = per_seg - pause
            if t < start:
                fr.append(0.0)
            else:
                fr.append(min(1.0, (t - start) / grow))
        return fr

    def render(frame):
        img = base.copy()
        if frame == 0:
            return img
        layer = base.copy()
        for (p0, p1), f in zip(segs, progress(frame)):
            if f <= 0:
                continue
            x = p0[0] + (p1[0] - p0[0]) * f
            y = p0[1] + (p1[1] - p0[1]) * f
            cv2.line(layer, (int(round(p0[0])), int(round(p0[1]))),
                     (int(round(x)), int(round(y))), RED, THICKNESS, cv2.LINE_AA)
        img[~protect] = layer[~protect]
        return img

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-sws_flags", "accurate_rnd+full_chroma_int",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        proc.stdin.write(render(i).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
