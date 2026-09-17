#!/usr/bin/env python3
"""Generate the block-stacking video: move CURRENT stacks (left) to match TARGET (right)."""
import os, subprocess, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 120

BG = np.array([245, 245, 250], np.uint8)
BLOCK_W, BLOCK_H = 99, 64             # sprite width, vertical pitch
BASE_Y = 824                          # bottom border row of a grounded block
LEFT_COLS = [79, 207, 335]            # left x of each CURRENT stack column
RIGHT_COLS = [719, 847, 975]          # left x of each TARGET column (3rd is empty)
TARGET_COL_X = [591, 719, 847]        # actual target columns (from base bars at 582,710,838)
COLORS = {"P": (155, 89, 182), "B": (52, 152, 219), "G": (46, 204, 113),
          "O": (230, 126, 34), "R": (231, 76, 60)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TEXT_CENTER = (512, 994)
TEXT_BOX = (400, 977, 624, 1012)      # x0,y0,x1,y1 region to clear before redrawing counter
LIFT_Y = 560                          # top of a block while travelling


def block_top(k):
    """Top row (cap row) of the sprite for a block at height index k (0 = on ground)."""
    return BASE_Y - BLOCK_H * (k + 1) - 1


def read_stacks(img, cols):
    """Detect stacks (bottom -> top letters) by sampling block body colour."""
    stacks = []
    for x0 in cols:
        s = []
        for k in range(8):
            y = block_top(k) + 20
            if y < 0:
                break
            px = tuple(int(v) for v in img[y + 12, x0 + 12])
            hit = [c for c, rgb in COLORS.items() if sum(abs(a - b) for a, b in zip(px, rgb)) < 40]
            if not hit:
                break
            s.append(hit[0])
        stacks.append(s)
    return stacks


def solve(start, goal):
    """BFS over stack configurations; a move takes the top block of one stack onto another."""
    start = tuple(tuple(s) for s in start)
    goal = tuple(tuple(s) for s in goal)
    prev = {start: None}
    q = collections.deque([start])
    while q:
        st = q.popleft()
        if st == goal:
            break
        for i, src in enumerate(st):
            if not src:
                continue
            for j in range(len(st)):
                if i == j:
                    continue
                ns = [list(s) for s in st]
                ns[j].append(ns[i].pop())
                ns = tuple(tuple(s) for s in ns)
                if ns not in prev:
                    prev[ns] = (st, (i, j))
                    q.append(ns)
    moves = []
    st = goal
    while prev[st] is not None:
        st, mv = prev[st]
        moves.append(mv)
    return moves[::-1]


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    cur = read_stacks(first, LEFT_COLS)
    tgt = read_stacks(first, TARGET_COL_X)
    moves = solve(cur, tgt)
    print("current:", cur, "target:", tgt, "moves:", moves)

    # Extract one sprite per block (66 rows: cap row + 64 body rows + bottom border row).
    sprites = {}
    for ci, stack in enumerate(cur):
        x0 = LEFT_COLS[ci]
        for k, letter in enumerate(stack):
            y0 = block_top(k)
            sprites[letter] = first[y0:y0 + BLOCK_H + 2, x0:x0 + BLOCK_W].copy()
    # Normalise the cap row (only a top block shows it) using a real top block's cap row.
    top_letter = cur[0][-1]
    cap = first[block_top(len(cur[0]) - 1), LEFT_COLS[0]:LEFT_COLS[0] + BLOCK_W].copy()
    for letter in sprites:
        sprites[letter][0] = cap

    # Static background: first frame with the CURRENT blocks and move counter erased.
    base = first.copy()
    base[block_top(7):BASE_Y + 1, LEFT_COLS[0] - 2:LEFT_COLS[-1] + BLOCK_W + 2] = BG
    base[TEXT_BOX[1]:TEXT_BOX[3], TEXT_BOX[0]:TEXT_BOX[2]] = BG
    font = ImageFont.truetype(FONT, 32)

    def paste(img, sprite, x, y):
        h, w = sprite.shape[:2]
        img[y:y + h, x:x + w] = sprite

    def render(stacks, moving=None, n_moves=0):
        img = base.copy()
        for ci, stack in enumerate(stacks):
            for k, letter in enumerate(stack):
                paste(img, sprites[letter], LEFT_COLS[ci], block_top(k))
        if moving is not None:
            letter, x, y = moving
            paste(img, sprites[letter], int(round(x)), int(round(y)))
        pil = Image.fromarray(img)
        ImageDraw.Draw(pil).text(TEXT_CENTER, f"Moves: {n_moves}", font=font,
                                 fill=(50, 50, 50), anchor="mm")
        return np.array(pil)

    # Timeline: hold, then equal time per move, then hold at the end.
    hold_start, hold_end = 6, 10
    n_mv = len(moves)
    span = N_FRAMES - hold_start - hold_end
    bounds = [hold_start + round(span * i / n_mv) for i in range(n_mv + 1)]

    frames = []
    state = [list(s) for s in cur]
    done = 0
    for f in range(N_FRAMES):
        if f < hold_start or f >= bounds[-1]:
            frames.append(render(state, None, done))
            continue
        mi = max(i for i in range(n_mv) if bounds[i] <= f)
        src, dst = moves[mi]
        f0, f1 = bounds[mi], bounds[mi + 1]
        t = (f - f0) / (f1 - f0)
        letter = state[src][-1]
        rest = [list(s) for s in state]
        rest[src].pop()
        sx, sy = LEFT_COLS[src], block_top(len(state[src]) - 1)
        dx, dy = LEFT_COLS[dst], block_top(len(state[dst]))
        # Phases: lift (0-0.3), travel (0.3-0.7), drop (0.7-1.0).
        if t < 0.3:
            x, y = sx, sy + (LIFT_Y - sy) * ease(t / 0.3)
        elif t < 0.7:
            x, y = sx + (dx - sx) * ease((t - 0.3) / 0.4), LIFT_Y
        else:
            x, y = dx, LIFT_Y + (dy - LIFT_Y) * ease((t - 0.7) / 0.3)
        frames.append(render(rest, (letter, x, y), done))
        if f == f1 - 1:                 # block lands: commit the move
            state = rest
            state[dst].append(letter)
            done += 1

    assert np.array_equal(frames[0], first), "first frame mismatch"
    assert [list(s) for s in read_stacks(frames[-1], LEFT_COLS)] == tgt, "final state mismatch"

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "6", "-preset", "slow", "-tune", "stillimage", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
