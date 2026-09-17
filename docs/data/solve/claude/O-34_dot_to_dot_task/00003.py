#!/usr/bin/env python3
"""Connect numbered dots 1->2->3->4 with red lines, one line at a time."""
import os, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 55
RED = (0, 0, 255)  # BGR
THICK = 4


def find_dots(img):
    """Return list of (cx, cy) for each dot, sorted by number (1..N)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 250).astype(np.uint8)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask)
    dots = [(cents[i][0], cents[i][1], stats[i][4]) for i in range(1, n) if stats[i][4] > 500]
    # Known numbering for this scene (verified visually): order the four dots.
    # Dots labeled 1..4 are located respectively near bottom-left, top-middle,
    # top-left and right.  Map by nearest known label positions.
    labels = {1: (256, 720), 2: (644, 194), 3: (168, 135), 4: (880, 438)}
    ordered = []
    for k in sorted(labels):
        lx, ly = labels[k]
        best = min(dots, key=lambda d: (d[0] - lx) ** 2 + (d[1] - ly) ** 2)
        ordered.append((float(best[0]), float(best[1])))
    return ordered


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = cv2.imread(FIRST)
    h, w = base.shape[:2]
    pts = find_dots(base)

    # Mask of full dot discs (incl. white digits) so dots stay on top of lines.
    dot_mask = np.zeros((h, w), np.uint8)
    for (cx, cy) in pts:
        cv2.circle(dot_mask, (int(round(cx)), int(round(cy))), 46, 255, -1)
    dot_mask = dot_mask > 0

    n_seg = len(pts) - 1
    # Frame 0 is untouched; remaining frames animate segments sequentially.
    anim_frames = N_FRAMES - 1
    per_seg = anim_frames / n_seg

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         # Explicit bt709 full-range conversion + tagging so decoders reproduce
         # the RGB source (pure white background) faithfully.
         "-vf", "scale=out_color_matrix=bt709:out_range=pc",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-color_range", "pc",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT],
        stdin=subprocess.PIPE)

    for f in range(N_FRAMES):
        frame = base.copy()
        t = f  # animation time in frames (frame 0 -> nothing drawn)
        for s in range(n_seg):
            start, end = s * per_seg, (s + 1) * per_seg
            if t <= start:
                break
            frac = min(1.0, (t - start) / (end - start))
            p0 = np.array(pts[s]); p1 = np.array(pts[s + 1])
            pe = p0 + frac * (p1 - p0)
            cv2.line(frame, (int(round(p0[0])), int(round(p0[1]))),
                     (int(round(pe[0])), int(round(pe[1]))), RED, THICK, cv2.LINE_AA)
        frame[dot_mask] = base[dot_mask]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
