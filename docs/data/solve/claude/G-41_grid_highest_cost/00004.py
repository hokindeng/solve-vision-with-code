#!/usr/bin/env python3
"""Animate Pac-Man along the maximum-cost monotone (right/down) path on a 4x4 grid."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 91
CELL = 256
PAC_R = 64            # outer radius (incl. 2px black outline)
PAC_DX, PAC_DY = 128, 148   # Pac-Man centre offset inside a cell (as in first frame)
YELLOW, BLACK, WHITE, GREEN = (255, 255, 0), (0, 0, 0), (255, 255, 255), (76, 175, 80)

# Costs read from first_frame.png (start cell cost is irrelevant: it is never "entered").
COSTS = [[0, 50, 10, 30],
         [20, 10, 50, 20],
         [50, 30, 10, 30],
         [20, 10, 10, 50]]
START, GOAL = (0, 0), (3, 3)


def best_path():
    """DP over right/down moves maximising the sum of entered-cell costs."""
    n = 4
    dp = [[None] * n for _ in range(n)]
    prev = [[None] * n for _ in range(n)]
    dp[0][0] = 0
    for r in range(n):
        for c in range(n):
            if (r, c) == (0, 0):
                continue
            cands = []
            if r > 0:
                cands.append((dp[r - 1][c], (r - 1, c)))
            if c > 0:
                cands.append((dp[r][c - 1], (r, c - 1)))
            v, p = max(cands, key=lambda t: t[0])   # tie -> first (from above)
            dp[r][c] = v + COSTS[r][c]
            prev[r][c] = p
    path, cur = [], GOAL
    while cur is not None:
        path.append(cur)
        cur = prev[cur[0]][cur[1]]
    return path[::-1], dp[GOAL[0]][GOAL[1]]


def pac_center(r, c):
    return c * CELL + PAC_DX, r * CELL + PAC_DY


def draw_pacman(img, cx, cy):
    d = ImageDraw.Draw(img)
    d.ellipse([cx - PAC_R, cy - PAC_R, cx + PAC_R, cy + PAC_R], fill=YELLOW, outline=BLACK, width=2)
    d.polygon([(cx, cy), (cx + 44, cy - 20), (cx + 44, cy + 20)], fill=WHITE)


def main():
    first = Image.open('/app/first_frame.png').convert('RGB')
    # Background: first frame with Pac-Man erased (it sits fully inside the green start cell).
    bg = first.copy()
    sx, sy = pac_center(*START)
    ImageDraw.Draw(bg).rectangle([sx - PAC_R, sy - PAC_R, sx + PAC_R, sy + PAC_R], fill=GREEN)

    path, total = best_path()
    print('path:', path, 'total cost:', total)
    steps = len(path) - 1
    per_step = (N_FRAMES - 1) // steps          # 15 frames per move
    move_frames = per_step * steps

    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-r', str(FPS),
         '/app/output/video.mp4'], stdin=subprocess.PIPE)

    for f in range(N_FRAMES):
        if f == 0:
            frame = first.copy()
        else:
            t = min(f, move_frames)
            k = min((t - 1) // per_step, steps - 1)
            u = (t - k * per_step) / per_step
            (x0, y0), (x1, y1) = pac_center(*path[k]), pac_center(*path[k + 1])
            cx, cy = round(x0 + (x1 - x0) * u), round(y0 + (y1 - y0) * u)
            frame = bg.copy()
            draw_pacman(frame, cx, cy)
        ff.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    print('wrote /app/output/video.mp4')


if __name__ == '__main__':
    main()
