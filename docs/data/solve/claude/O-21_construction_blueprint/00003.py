#!/usr/bin/env python3
"""Generate the candidate-checking / gap-filling video from first_frame.png."""
import subprocess
import numpy as np
import cv2
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 99

BG = (245, 245, 245)
CELL_FILL = (105, 105, 105)
CELL_EDGE = (50, 100, 140)
PANEL_BG = (230, 230, 230)
PANEL_EDGE = (180, 180, 180)
GREEN = (60, 170, 80)
GREEN_BG = (205, 235, 210)
RED = (215, 60, 60)
RED_BG = (245, 205, 205)
HILITE = (255, 160, 0)

# structure geometry
S_CELL, S_PITCH = 51, 52
GAP_ANCHOR = (486, 383)                 # top-left of first gap cell
GAP_CELLS = [(0, 0), (1, 0)]
DASH_BBOX = (486, 383, 589, 434)        # inclusive bbox of the red dashed outline

# candidate panels (inclusive interior bounds) and their pieces
C_CELL, C_PITCH = 37, 38
PANELS = [(51, 830, 234, 1013), (297, 830, 480, 1013),
          (543, 830, 726, 1013), (789, 830, 972, 1013)]
PIECES = [
    dict(cells=[(0, 0), (0, 1)], origin=(123, 883)),
    dict(cells=[(0, 0), (1, 0), (2, 0)], origin=(331, 902)),
    dict(cells=[(0, 0)], origin=(615, 902)),
    dict(cells=[(0, 0), (1, 0)], origin=(842, 902)),
]
MATCH = 3

# timeline
T0 = 1              # first animated frame
PER = 16            # frames per candidate
MOVE_START = T0 + PER * 4 + 2
MOVE_END = 93       # piece lands here; hold afterwards


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def draw_cell(img, x, y, size, alpha=1.0):
    """Draw one structure-style cell (1px blue edge, dark fill) top-left at (x, y)."""
    x, y, size = int(round(x)), int(round(y)), int(round(size))
    patch = img[y:y + size, x:x + size].astype(np.float32)
    cell = np.empty_like(patch)
    cell[:] = CELL_FILL
    cell[0, :] = cell[-1, :] = cell[:, 0] = cell[:, -1] = CELL_EDGE
    img[y:y + size, x:x + size] = (patch * (1 - alpha) + cell * alpha).round().astype(np.uint8)


def draw_piece(img, cells, ox, oy, size, pitch, alpha=1.0):
    for cx, cy in cells:
        draw_cell(img, ox + cx * pitch, oy + cy * pitch, size, alpha)


def highlight(img, i):
    x0, y0, x1, y1 = PANELS[i]
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), HILITE, 3)


def judge(img, base, i, ok):
    x0, y0, x1, y1 = PANELS[i]
    tint, edge = (GREEN_BG, GREEN) if ok else (RED_BG, RED)
    region = base[y0 - 1:y1 + 2, x0 - 1:x1 + 2]
    out = img[y0 - 1:y1 + 2, x0 - 1:x1 + 2]
    out[np.all(region == PANEL_BG, axis=2)] = tint
    out[np.all(region == PANEL_EDGE, axis=2)] = edge
    # mark in the top-right corner of the panel
    cx, cy, r = x1 - 24, y0 + 22, 11
    if ok:
        pts = np.array([(cx - r, cy), (cx - r // 3, cy + r - 2), (cx + r + 2, cy - r + 1)], np.int32)
        cv2.polylines(img, [pts], False, edge, 4, cv2.LINE_AA)
    else:
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), edge, 4, cv2.LINE_AA)
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), edge, 4, cv2.LINE_AA)


def render(f, base):
    img = base.copy()
    # persistent verdicts for candidates already judged
    for i in range(4):
        start = T0 + PER * i
        if f >= start + 10:
            judge(img, base, i, i == MATCH)
    if f < MOVE_START:
        i = min((f - T0) // PER, 3) if f >= T0 else -1
        if i >= 0:
            k = f - (T0 + PER * i)
            highlight(img, i)
            # preview ghost: fade in over frames 2..5, hold, fade out during judging
            if 2 <= k < 10:
                a = 0.55 * ease((k - 1) / 4)
            elif k >= 10:
                a = 0.55 * (1 - ease((k - 9) / 5))
            else:
                a = 0.0
            if a > 0:
                draw_piece(img, PIECES[i]['cells'], GAP_ANCHOR[0], GAP_ANCHOR[1],
                           S_CELL, S_PITCH, a)
    else:
        t = ease((f - MOVE_START) / (MOVE_END - MOVE_START))
        p = PIECES[MATCH]
        if t >= 1.0:
            x0, y0, x1, y1 = DASH_BBOX
            img[y0:y1 + 1, x0:x1 + 1] = BG
            draw_piece(img, p['cells'], GAP_ANCHOR[0], GAP_ANCHOR[1], S_CELL, S_PITCH)
        else:
            ox = p['origin'][0] + (GAP_ANCHOR[0] - p['origin'][0]) * t
            oy = p['origin'][1] + (GAP_ANCHOR[1] - p['origin'][1]) * t
            size = C_CELL + (S_CELL - C_CELL) * t
            pitch = C_PITCH + (S_PITCH - C_PITCH) * t
            draw_piece(img, p['cells'], ox, oy, size, pitch)
    return img


def main():
    base = np.array(Image.open('/app/first_frame.png').convert('RGB'))
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '10', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', '/app/output/video.mp4']
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        proc.stdin.write(render(f, base).tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == '__main__':
    main()
