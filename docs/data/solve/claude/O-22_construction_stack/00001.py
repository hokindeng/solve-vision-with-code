#!/usr/bin/env python3
"""Animate rearranging the CURRENT stacks into the TARGET layout.

Blocks are cut out of first_frame.png as sprites and re-pasted, so the
appearance of every block, the platforms, labels and TARGET side are
pixel-identical to the first frame. Only the left blocks and the
"Moves" counter change.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/app"
FIRST = f"{ROOT}/first_frame.png"
OUT = f"{ROOT}/output/video.mp4"
FPS, W, H, N_FRAMES = 16, 1024, 1024, 150

BG = (245, 245, 250)
TEXT_COLOR = (50, 50, 50)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
TEXT_POS = (431, 975)

# Geometry measured from first_frame.png (left/CURRENT side)
STACK_X = [79, 207, 335]      # left edge of block in each stack
BLOCK_W, BLOCK_H = 99, 65     # sprite size (borders shared between levels)
LEVEL_PITCH = 64
BASE_Y = 824                  # bottom border row of the level-0 block
LIFT_Y = 380                  # top of a carried block while travelling

def block_top(level):
    return BASE_Y - LEVEL_PITCH * (level + 1)

# Initial state (bottom -> top) and the move plan (from_stack, to_stack).
STACKS = [["O", "B"], ["R"], ["Y", "P"]]
MOVES = [(2, 0), (2, 0), (1, 2), (0, 1), (0, 2), (0, 1), (0, 2), (1, 0)]
# result: [B], [Y], [R, P, O]  == TARGET


def smoothstep(t):
    return t * t * (3 - 2 * t)


def main():
    base = Image.open(FIRST).convert("RGB")
    arr = np.array(base)

    # Cut sprites for each block from its position in the first frame.
    sprites = {}
    for s, stack in enumerate(STACKS):
        for lvl, name in enumerate(stack):
            x0, y0 = STACK_X[s], block_top(lvl)
            sprites[name] = base.crop((x0, y0, x0 + BLOCK_W, y0 + BLOCK_H))

    # Background: first frame with left blocks and the counter text erased.
    bgim = base.copy()
    d = ImageDraw.Draw(bgim)
    d.rectangle((0, 300, 500, BASE_Y), fill=BG)
    d.rectangle((400, 978, 640, 1015), fill=BG)

    def render(stacks, carried, moves_done):
        im = bgim.copy()
        for s, stack in enumerate(stacks):
            for lvl, name in enumerate(stack):
                im.paste(sprites[name], (STACK_X[s], block_top(lvl)))
        if carried is not None:
            name, x, y = carried
            im.paste(sprites[name], (int(round(x)), int(round(y))))
        ImageDraw.Draw(im).text(TEXT_POS, f"Moves: {moves_done}", font=FONT, fill=TEXT_COLOR)
        return np.array(im)

    frames = []
    stacks = [list(s) for s in STACKS]
    HOLD_START, HOLD_END = 6, 8
    per_move = (N_FRAMES - HOLD_START - HOLD_END) // len(MOVES)  # 17
    up_n, across_n = 5, 6
    down_n = per_move - up_n - across_n

    frames.append(arr.copy())  # exact first frame
    for _ in range(HOLD_START - 1):
        frames.append(render(stacks, None, 0))

    for mi, (src, dst) in enumerate(MOVES):
        name = stacks[src].pop()
        x0, y0 = STACK_X[src], block_top(len(stacks[src]))
        x1, y1 = STACK_X[dst], block_top(len(stacks[dst]))
        # lift
        for i in range(up_n):
            t = smoothstep((i + 1) / up_n)
            frames.append(render(stacks, (name, x0, y0 + (LIFT_Y - y0) * t), mi))
        # travel
        for i in range(across_n):
            t = smoothstep((i + 1) / across_n)
            frames.append(render(stacks, (name, x0 + (x1 - x0) * t, LIFT_Y), mi))
        # drop
        for i in range(down_n):
            t = smoothstep((i + 1) / down_n)
            if i == down_n - 1:
                stacks[dst].append(name)
                frames.append(render(stacks, None, mi + 1))
            else:
                frames.append(render(stacks, (name, x1, LIFT_Y + (y1 - LIFT_Y) * t), mi))

    assert stacks == [["B"], ["Y"], ["R", "P", "O"]], stacks
    while len(frames) < N_FRAMES:
        frames.append(render(stacks, None, len(MOVES)))
    frames = frames[:N_FRAMES]

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print(f"wrote {OUT}: {len(frames)} frames")


if __name__ == "__main__":
    main()
