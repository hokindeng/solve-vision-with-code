#!/usr/bin/env python3
"""Rearrange the CURRENT stacks (left) into the TARGET layout (right).

Blocks are cut as pixel sprites out of first_frame.png, so their appearance is
identical to the source. Only the left-side block area and the "Moves: N"
counter change; every other pixel is copied from the first frame.
"""
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/app"
FIRST = f"{ROOT}/first_frame.png"
OUT = f"{ROOT}/output/video.mp4"
FPS = 16
N_FRAMES = 135

BG = np.array([245, 245, 250], dtype=np.uint8)
DARK = (50, 50, 50)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Geometry measured from first_frame.png
BLOCK_W = 99          # x0 .. x0+98 inclusive
BLOCK_H = 65          # y0 .. y0+64 inclusive (adjacent blocks share the outline row)
STACK_X0 = [79, 207, 335]          # left (CURRENT) stacks
FLOOR_Y1 = 824                     # bottom outline row of the bottom block
TRAVEL_Y0 = 470                    # top of a block while carried sideways
COUNTER_ANCHOR = (512, 1005)       # 'ms' anchor of "Moves: N"

START = (("P", "Y", "G"), ("R", "B"), ())
GOAL = (("R", "P"), ("G", "Y"), ("B",))


def block_y0(k):
    """Top row of the block at height index k (0 = bottom)."""
    return FLOOR_Y1 - BLOCK_H + 1 - 64 * k


def plan_moves(start, goal):
    """Shortest sequence of (src, dst) top-block moves via BFS."""
    prev = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s == goal:
            break
        for i in range(3):
            if not s[i]:
                continue
            for j in range(3):
                if i == j:
                    continue
                t = [list(x) for x in s]
                t[j].append(t[i].pop())
                t = tuple(tuple(x) for x in t)
                if t not in prev:
                    prev[t] = (s, i, j)
                    q.append(t)
    moves = []
    s = goal
    while prev[s] is not None:
        s, i, j = prev[s]
        moves.append((i, j))
    return moves[::-1]


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = first.shape

    # --- cut block sprites from the first frame -------------------------------
    sprites = {}
    for s, stack in enumerate(START):
        for k, name in enumerate(stack):
            x0, y0 = STACK_X0[s], block_y0(k)
            sprites[name] = first[y0:y0 + BLOCK_H, x0:x0 + BLOCK_W].copy()

    # --- background: first frame with left blocks and counter removed ----------
    base = first.copy()
    top_clear = block_y0(4) - 4
    base[top_clear:FLOOR_Y1 + 2, STACK_X0[0] - 4:STACK_X0[2] + BLOCK_W + 4] = BG
    # counter text area (the divider ends above it; verify and keep it intact)
    txt_rows = slice(976, 1012)
    txt_cols = slice(420, 604)
    assert not (first[txt_rows, 512] == 200).all(axis=1).any()
    base[txt_rows, txt_cols] = BG

    font = ImageFont.truetype(FONT, 32)

    def paste(canvas, name, x0, y0):
        canvas[y0:y0 + BLOCK_H, x0:x0 + BLOCK_W] = sprites[name]
        # tiny rounded-corner pixels just outside the rectangle (as in the source)
        for yy in (y0 - 1, y0 + BLOCK_H):
            for xx in (x0 + 1, x0 + BLOCK_W - 2):
                if 0 <= yy < H:
                    canvas[yy, xx] = DARK

    def render(stacks, moving, counter):
        """moving = (name, x0, y0) or None."""
        canvas = base.copy()
        for s, stack in enumerate(stacks):
            for k, name in enumerate(stack):
                paste(canvas, name, STACK_X0[s], block_y0(k))
        if moving is not None:
            paste(canvas, *moving)
        img = Image.fromarray(canvas)
        ImageDraw.Draw(img).text(COUNTER_ANCHOR, f"Moves: {counter}", fill=DARK,
                                 font=font, anchor="ms")
        return np.array(img)

    # --- timeline --------------------------------------------------------------
    moves = plan_moves(START, GOAL)
    n_moves = len(moves)                       # 7
    LIFT, TRAVEL, DROP, SETTLE = 4, 8, 4, 1
    per_move = LIFT + TRAVEL + DROP + SETTLE   # 17
    intro = 3
    outro = N_FRAMES - intro - per_move * n_moves
    assert outro >= 1, outro

    frames = []
    frames.append(first.copy())                # exact first frame
    stacks = [list(x) for x in START]
    for _ in range(intro - 1):
        frames.append(render(stacks, None, 0))

    for m, (src, dst) in enumerate(moves):
        name = stacks[src].pop()
        k_src, k_dst = len(stacks[src]), len(stacks[dst])
        xs, xd = STACK_X0[src], STACK_X0[dst]
        ys, yd = block_y0(k_src), block_y0(k_dst)
        for f in range(per_move):
            if f < LIFT:
                t = ease((f + 1) / LIFT)
                pos = (xs, round(ys + (TRAVEL_Y0 - ys) * t))
            elif f < LIFT + TRAVEL:
                t = ease((f - LIFT + 1) / TRAVEL)
                pos = (round(xs + (xd - xs) * t), TRAVEL_Y0)
            elif f < LIFT + TRAVEL + DROP:
                t = ease((f - LIFT - TRAVEL + 1) / DROP)
                pos = (xd, round(TRAVEL_Y0 + (yd - TRAVEL_Y0) * t))
            else:
                pos = None
            if pos is None:
                stacks[dst].append(name)
                frames.append(render(stacks, None, m + 1))
            else:
                frames.append(render(stacks, (name, pos[0], pos[1]), m))
    assert [tuple(s) for s in stacks] == list(GOAL)
    while len(frames) < N_FRAMES:
        frames.append(render(stacks, None, n_moves))
    frames = frames[:N_FRAMES]

    # --- encode ----------------------------------------------------------------
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    Image.fromarray(frames[-1]).save(f"{ROOT}/output/last_frame.png")
    print(f"wrote {OUT}: {len(frames)} frames, {n_moves} moves: {moves}")


if __name__ == "__main__":
    main()
