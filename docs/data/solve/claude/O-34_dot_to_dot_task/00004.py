#!/usr/bin/env python3
"""Connect numbered dots in order with red lines, one at a time, and render an mp4."""
import os, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 55
RED = (0, 0, 255)  # BGR
THICKNESS = 6


def find_dots(img):
    """Return list of (x, y, r) for the coloured discs, plus their pixel mask."""
    mask = (np.abs(img.astype(int) - 255).sum(2) > 30).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    dots = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 500:
            x, y = cent[i]
            r = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / 2.0
            dots.append((x, y, r))
    return dots


def main():
    base = cv2.imread(FIRST)
    h, w = base.shape[:2]
    dots = find_dots(base)
    assert len(dots) == 4, dots

    # Known numbering for this scene (dot label -> position), matched to detected discs.
    labelled = {1: (445, 861), 2: (193, 533), 3: (523, 503), 4: (534, 163)}
    order = []
    for k in sorted(labelled):
        lx, ly = labelled[k]
        x, y, r = min(dots, key=lambda d: (d[0] - lx) ** 2 + (d[1] - ly) ** 2)
        order.append((x, y, r))

    # Mask of the dots (disc + outline) so that lines are drawn underneath them.
    dot_mask = np.zeros((h, w), np.uint8)
    for x, y, r in order:
        cv2.circle(dot_mask, (int(round(x)), int(round(y))), int(round(r)) + 1, 255, -1)
    dot_mask = dot_mask.astype(bool)

    segments = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    n_seg = len(segments)
    # Frame 0 is untouched; the final few frames hold the finished drawing.
    hold = 4
    anim_frames = N_FRAMES - 1 - hold
    per_seg = anim_frames / n_seg

    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        if f > 0:
            t_total = min((f) / per_seg, n_seg)  # segments fully/partially drawn
            for s, (a, b) in enumerate(segments):
                if t_total <= s:
                    break
                t = min(1.0, t_total - s)
                # ease-in-out for a natural pen stroke
                te = t * t * (3 - 2 * t)
                x0, y0 = a[0], a[1]
                x1 = x0 + (b[0] - x0) * te
                y1 = y0 + (b[1] - y0) * te
                cv2.line(img, (int(round(x0)), int(round(y0))),
                         (int(round(x1)), int(round(y1))), RED, THICKNESS, cv2.LINE_AA)
            img[dot_mask] = base[dot_mask]
        frames.append(img)

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:"
                  "flags=accurate_rnd+full_chroma_int,format=yuv420p",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
