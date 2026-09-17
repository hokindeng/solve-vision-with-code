#!/usr/bin/env python3
"""Animate the CURRENT stacks (left) being rearranged to match the TARGET (right).

Everything is composed from pixels of first_frame.png: block sprites are cut out of
the first frame and re-pasted, the Moves counter is re-rendered with the same font.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 120
BG = (245, 245, 250)
DIVIDER = (200, 200, 200)
TEXT_COLOR = (50, 50, 50)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)

# Block geometry measured from first_frame.png
BW, BH = 99, 65          # sprite size (includes 2px outline)
PITCH_X, PITCH_Y = 128, 64
LEFT_X0 = 79             # left edge of left stack 0
GROUND_Y = 824           # y of the bottom outline row of a ground-level block
CARRY_Y = 400            # top y of a block while being carried

# Initial state (bottom -> top) of the CURRENT stacks and where each sprite is
CURRENT = [["R", "O"], ["Y", "B"], ["G"]]
SPRITE_POS = {"O": (79, 696), "R": (79, 760), "B": (207, 696), "Y": (207, 760), "G": (335, 760)}

# Target: [[], [O, Y, B, R], [G]]  -> minimal 6-move plan (from_stack, to_stack)
MOVES = [(1, 2), (1, 2), (0, 1), (2, 1), (2, 1), (0, 1)]


def stack_x(i):
    return LEFT_X0 + PITCH_X * i


def level_y(k):
    return GROUND_Y - PITCH_Y * (k + 1)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = np.array(Image.open(FIRST).convert("RGB"))

    # Sprites are 99x65; the rounded outline also has one corner pixel just above and below
    # the sprite (columns x+1 and x+97). Blocks are pasted bottom-up so an upper block
    # hides the top corner pixels of the block beneath it, as in the first frame.
    sprites = {k: first[y:y + BH, x:x + BW].copy() for k, (x, y) in SPRITE_POS.items()}

    def paste(frame, c, x, y):
        frame[y:y + BH, x:x + BW] = sprites[c]
        for cy in (y - 1, y + BH):
            frame[cy, x + 1] = frame[cy, x + BW - 2] = (50, 50, 50)

    # Static background: first frame with the left blocks and counter text removed.
    base = first.copy()
    base[60:GROUND_Y + 2, 60:450] = BG                 # left stacks area (blocks only)
    base[960:1016, 420:606] = BG                       # counter text area
    base[960:975, 512:514] = DIVIDER                   # divider continues under the text

    def render(stacks, carried, moves_done):
        frame = base.copy()
        for i, st in enumerate(stacks):
            for k, c in enumerate(st):
                x, y = stack_x(i), level_y(k)
                paste(frame, c, x, y)
        if carried is not None:
            c, x, y = carried
            paste(frame, c, x, y)
        img = Image.fromarray(frame)
        ImageDraw.Draw(img).text((512, 1005), f"Moves: {moves_done}", font=FONT,
                                 fill=TEXT_COLOR, anchor="ms")
        return np.array(img)

    # Timeline: frame 0 static, then 6 moves x 19 frames (lift 4, travel 8, drop 4, rest 3),
    # then hold the finished state.
    LIFT, TRAVEL, DROP, REST = 4, 8, 4, 3
    PER_MOVE = LIFT + TRAVEL + DROP + REST
    frames = [render(CURRENT, None, 0)]
    stacks = [list(s) for s in CURRENT]
    done = 0
    for src, dst in MOVES:
        color = stacks[src].pop()
        x0, y0 = stack_x(src), level_y(len(stacks[src]))
        x1, y1 = stack_x(dst), level_y(len(stacks[dst]))
        for f in range(1, LIFT + 1):
            y = round(y0 + (CARRY_Y - y0) * ease(f / LIFT))
            frames.append(render(stacks, (color, x0, y), done))
        for f in range(1, TRAVEL + 1):
            x = round(x0 + (x1 - x0) * ease(f / TRAVEL))
            frames.append(render(stacks, (color, x, CARRY_Y), done))
        for f in range(1, DROP + 1):
            y = round(CARRY_Y + (y1 - CARRY_Y) * ease(f / DROP))
            if f == DROP:
                stacks[dst].append(color)
                done += 1
                frames.append(render(stacks, None, done))
            else:
                frames.append(render(stacks, (color, x1, y), done))
        for _ in range(REST):
            frames.append(render(stacks, None, done))
    while len(frames) < N_FRAMES:
        frames.append(render(stacks, None, done))
    frames = frames[:N_FRAMES]
    assert stacks == [[], ["O", "Y", "B", "R"], ["G"]], stacks

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-sws_flags", "lanczos+accurate_rnd+full_chroma_int",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, "ffmpeg failed"

    # Save a couple of frames for inspection.
    Image.fromarray(frames[0]).save(os.path.join(OUT_DIR, "frame_first.png"))
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "frame_last.png"))
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
