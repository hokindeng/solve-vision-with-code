#!/usr/bin/env python3
"""Generate the puzzle-solving video from first_frame.png."""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N = 16, 99
W = H = 1024

BG = (245, 245, 245)
ORANGE = (200, 100, 0)
BLUE = (50, 100, 140)
BOX_FILL = (230, 230, 230)
BOX_BORDER = (180, 180, 180)
GREEN = (40, 170, 70)
RED = (220, 50, 50)
GREEN_FILL = (205, 235, 210)
RED_FILL = (245, 210, 210)
HILITE = (30, 110, 200)

# ---- scene geometry (measured from first_frame.png) ----
GAP_CELL, GAP_PITCH = 50, 52          # main structure cells: rect [x0, x0+50], pitch 52
GAP_ORIGIN = (382, 227)               # top-left of gap bounding box
GAP_CELLS = [(1, 0), (0, 1), (1, 1), (1, 2)]   # (col,row) inside the gap bbox

CAND_CELL, CAND_PITCH = 36, 37        # candidate cells: rect [x0, x0+36], pitch 37
BOXES = [(50, 829, 235, 1014), (296, 829, 481, 1014),
         (542, 829, 727, 1014), (788, 829, 973, 1014)]
# candidate shapes: (origin top-left, list of (col,row))
CANDS = [
    ((85, 864), [(1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]),
    ((350, 883), [(0, 0), (1, 0), (1, 1)]),
    ((596, 864), [(0, 0), (0, 1), (1, 1), (0, 2)]),
    ((842, 864), [(1, 0), (0, 1), (1, 1), (1, 2)]),
]
MATCH = 3  # index of the candidate that fits the gap exactly


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def draw_piece(img, origin, cells, cell, pitch, alpha=1.0):
    """Draw a polyomino as orange squares with blue borders; alpha<1 gives a ghost."""
    ox, oy = origin
    if alpha >= 1.0:
        d = ImageDraw.Draw(img)
        for c, r in cells:
            x0, y0 = int(round(ox + c * pitch)), int(round(oy + r * pitch))
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=ORANGE, outline=BLUE, width=1)
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * alpha)
    for c, r in cells:
        x0, y0 = int(round(ox + c * pitch)), int(round(oy + r * pitch))
        d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=ORANGE + (a,), outline=BLUE + (a,), width=1)
    base = img.convert("RGBA")
    base.alpha_composite(layer)
    return base.convert("RGB")


def highlight_box(img, box, color=HILITE):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=color, width=3)


def judge_box(img, box, ok):
    """Recolor a candidate box (fill + border) and stamp a check / cross in its corner."""
    arr = np.array(img)
    x0, y0, x1, y1 = box
    sub = arr[y0:y1 + 1, x0:x1 + 1]
    fill_mask = np.all(sub == BOX_FILL, axis=-1)
    sub[fill_mask] = GREEN_FILL if ok else RED_FILL
    arr[y0:y1 + 1, x0:x1 + 1] = sub
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    col = GREEN if ok else RED
    d.rectangle([x0, y0, x1, y1], outline=col, width=2)
    # mark in the top-right corner
    cx, cy, s = x1 - 24, y0 + 22, 11
    if ok:
        d.line([(cx - s, cy), (cx - s // 3, cy + s - 2), (cx + s, cy - s + 2)], fill=col, width=4)
    else:
        d.line([(cx - s, cy - s), (cx + s, cy + s)], fill=col, width=4)
        d.line([(cx - s, cy + s), (cx + s, cy - s)], fill=col, width=4)
    return img


def erase_gap(img):
    """Remove the dashed outline: clear only the gap cells (pitch-wide, 2px dashed border inside)."""
    d = ImageDraw.Draw(img)
    ox, oy = GAP_ORIGIN
    for c, r in GAP_CELLS:
        x0, y0 = ox + c * GAP_PITCH, oy + r * GAP_PITCH
        d.rectangle([x0, y0, x0 + GAP_PITCH - 1, y0 + GAP_PITCH - 1], fill=BG)


def place_final(img):
    erase_gap(img)
    draw_piece(img, GAP_ORIGIN, GAP_CELLS, GAP_CELL, GAP_PITCH)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")

    # ---- timeline ----
    # per-candidate: 4 highlight, 8 preview, 6 judge  = 18 frames, x4 = 72 (frames 1..72)
    # final move: frames 73..92 (20 frames), hold 93..98
    HL, PV, JD = 4, 8, 6
    PER = HL + PV + JD
    START = 1
    MOVE_START = START + 4 * PER      # 73
    MOVE_LEN = 20
    verdicts = [i == MATCH for i in range(4)]

    frames = []
    for f in range(N):
        img = base.copy()
        if f == 0:
            frames.append(img)
            continue

        # judged candidates so far
        cur = None
        if f < MOVE_START:
            k = (f - START) // PER
            ph = (f - START) % PER
            cur = k
            for i in range(k):
                img = judge_box(img, BOXES[i], verdicts[i])
            origin, cells = CANDS[k]
            if ph < HL:
                highlight_box(img, BOXES[k])
            elif ph < HL + PV:
                highlight_box(img, BOXES[k])
                t = (ph - HL + 1) / PV
                img = draw_piece(img, GAP_ORIGIN, cells, GAP_CELL, GAP_PITCH,
                                 alpha=0.25 + 0.4 * min(1.0, t * 2))
            else:
                img = judge_box(img, BOXES[k], verdicts[k])
                if verdicts[k]:
                    img = draw_piece(img, GAP_ORIGIN, cells, GAP_CELL, GAP_PITCH, alpha=0.65)
        else:
            for i in range(4):
                img = judge_box(img, BOXES[i], verdicts[i])
            origin, cells = CANDS[MATCH]
            t = ease((f - MOVE_START + 1) / MOVE_LEN)
            if t >= 1.0:
                place_final(img)
            else:
                ox = origin[0] + (GAP_ORIGIN[0] - origin[0]) * t
                oy = origin[1] + (GAP_ORIGIN[1] - origin[1]) * t
                cell = CAND_CELL + (GAP_CELL - CAND_CELL) * t
                pitch = CAND_PITCH + (GAP_PITCH - CAND_PITCH) * t
                img = draw_piece(img, (ox, oy), cells, cell, pitch)
        frames.append(img)

    # ---- encode ----
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-tune", "stillimage", "-x264-params", "keyint=16", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
