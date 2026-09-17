#!/usr/bin/env python3
"""Regenerate the shape-sorter video from first_frame.png.

Each colored card is lifted from the first frame as an exact pixel mask, the
background is restored underneath it, and the card is then slid (one at a
time, eased, integer-pixel steps) onto its matching outline on the right.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 78

CARD_COLORS = {
    "blue": (96, 165, 250),
    "red": (248, 113, 113),
    "cyan": (34, 211, 238),
}
OUTLINE_COLOR = (100, 116, 139)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), xs.max(), ys.min(), ys.max()


def main():
    frame0 = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = frame0.shape
    bg_color = frame0[5, 5].copy()

    # --- extract cards --------------------------------------------------
    cards = {}
    background = frame0.copy()
    for name, col in CARD_COLORS.items():
        mask = np.all(frame0 == np.array(col, dtype=np.uint8), axis=2)
        x0, x1, y0, y1 = bbox(mask)
        cards[name] = {
            "mask": mask[y0:y1 + 1, x0:x1 + 1],
            "color": np.array(col, dtype=np.uint8),
            "pos": np.array([x0, y0], dtype=float),
            "size": np.array([x1 - x0 + 1, y1 - y0 + 1]),
        }
        background[mask] = bg_color  # restore the staging area

    # --- find outline targets (right side of divider) -------------------
    omask = np.all(frame0 == np.array(OUTLINE_COLOR, dtype=np.uint8), axis=2)
    omask[:, :600] = False
    rows = np.nonzero(omask.any(axis=1))[0]
    # split into connected row groups
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    centers = []
    for g in groups:
        sub = omask.copy()
        sub[:g[0]] = False
        sub[g[-1] + 1:] = False
        x0, x1, y0, y1 = bbox(sub)
        centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    # outlines from top to bottom: square, diamond, circle
    order = ["blue", "red", "cyan"]
    targets = {}
    for name, (cx, cy) in zip(order, centers):
        sz = cards[name]["size"]
        targets[name] = np.array([round(cx - (sz[0] - 1) / 2.0),
                                  round(cy - (sz[1] - 1) / 2.0)], dtype=float)

    # --- schedule: three sequential slides with brief holds -------------
    # (start_frame, end_frame) inclusive-exclusive for each move
    moves = [("blue", 2, 26), ("red", 28, 52), ("cyan", 54, 76)]

    def ease(t):
        return t * t * (3 - 2 * t)  # smoothstep

    def card_pos(name, f):
        start = cards[name]["pos"]
        end = targets[name]
        for nm, s, e in moves:
            if nm != name:
                continue
            if f < s:
                return start
            if f >= e:
                return end
            t = ease((f - s) / float(e - 1 - s))
            return start + (end - start) * t
        return start

    def render(f):
        img = background.copy()
        for name in order:
            c = cards[name]
            x, y = np.round(card_pos(name, f)).astype(int)
            h, w = c["mask"].shape
            region = img[y:y + h, x:x + w]
            region[c["mask"]] = c["color"]
        return img

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [render(f) for f in range(N_FRAMES)]
    assert np.array_equal(frames[0], frame0), "first frame must match exactly"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT, "frames:", len(frames), "targets:", {k: v.tolist() for k, v in targets.items()})


if __name__ == "__main__":
    main()
