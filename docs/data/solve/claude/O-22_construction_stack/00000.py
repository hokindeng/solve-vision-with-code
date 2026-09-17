#!/usr/bin/env python3
"""Generate the block-stacking video: move blocks on the CURRENT side so it
matches the TARGET side, one top block per move, updating the move counter."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/app'
FIRST = f'{ROOT}/first_frame.png'
OUT = f'{ROOT}/output/video.mp4'
W = H = 1024
FPS = 16
N_FRAMES = 90

BG = np.array([245, 245, 250], dtype=np.uint8)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 32)

# Layout measured from first_frame.png
STACK_CX = [128, 256, 384]        # CURRENT stack centres (x)
SPR_W, SPR_H = 99, 67             # block sprite size
BASE_TOP = 759                    # top row of a block sitting on the platform
PITCH = 64                        # vertical distance between stacked blocks
HOVER_TOP = 540                   # top row of a carried block while travelling

first = np.array(Image.open(FIRST).convert('RGB'))


def crop(y, x):
    return first[y:y + SPR_H, x:x + SPR_W].copy()


# --- sprites, taken from the actual pixels of the first frame ---------------
spr_O = crop(759, 79)                       # free-standing block, clean edges
spr_G = crop(695, 335)                      # bottom two rows overlap block P
spr_G[65:67] = spr_O[65:67]                 # restore clean bottom edge
spr_P = crop(759, 335)                      # top row overlaps block G
spr_P[0] = spr_O[0]                         # restore clean top edge
SPRITES = {'O': spr_O, 'G': spr_G, 'P': spr_P}

# --- static background: first frame minus the movable blocks and counter ----
background = first.copy()
background[695:826, 79:434] = BG            # left-side blocks
background[978:1011, 420:606] = BG          # "Moves: N" label


def paste(canvas, spr, top, left):
    mask = np.any(spr != BG, axis=2)
    region = canvas[top:top + SPR_H, left:left + SPR_W]
    region[mask] = spr[mask]


def draw_label(canvas, moves):
    img = Image.fromarray(canvas)
    ImageDraw.Draw(img).text((512, 994), f'Moves: {moves}', font=FONT,
                             fill=(50, 50, 50), anchor='mm')
    return np.array(img)


def block_pos(stack, level):
    return BASE_TOP - PITCH * level, STACK_CX[stack] - SPR_W // 2


def render(stacks, moves, carried=None):
    canvas = background.copy()
    for si, stack in enumerate(stacks):
        for level, name in enumerate(stack):
            top, left = block_pos(si, level)
            paste(canvas, SPRITES[name], top, left)
    if carried is not None:
        name, top, left = carried
        paste(canvas, SPRITES[name], int(round(top)), int(round(left)))
    return draw_label(canvas, moves)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


# --- plan: CURRENT [O] [] [P,G]  ->  TARGET [] [P,G,O] [] --------------------
stacks = [['O'], [], ['P', 'G']]
MOVES = [(2, 0), (2, 1), (0, 1), (0, 1)]   # (from, to), top block each time

FRAMES_PER_MOVE = 20
LIFT, TRAVEL, DROP = 5, 10, 5
frames = [render(stacks, 0)]                # frame 0 == first_frame.png
moves_done = 0
for src, dst in MOVES:
    name = stacks[src].pop()
    y0, x0 = block_pos(src, len(stacks[src]))
    y1, x1 = block_pos(dst, len(stacks[dst]))
    for f in range(1, FRAMES_PER_MOVE + 1):
        if f <= LIFT:
            t = ease(f / LIFT)
            top, left = y0 + (HOVER_TOP - y0) * t, x0
        elif f <= LIFT + TRAVEL:
            t = ease((f - LIFT) / TRAVEL)
            top, left = HOVER_TOP, x0 + (x1 - x0) * t
        else:
            t = ease((f - LIFT - TRAVEL) / DROP)
            top, left = HOVER_TOP + (y1 - HOVER_TOP) * t, x1
        if f == FRAMES_PER_MOVE:
            stacks[dst].append(name)
            moves_done += 1
            frames.append(render(stacks, moves_done))
        else:
            frames.append(render(stacks, moves_done, (name, top, left)))
while len(frames) < N_FRAMES:                # hold the finished state
    frames.append(frames[-1].copy())
frames = frames[:N_FRAMES]

# --- encode -----------------------------------------------------------------
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
       '-c:v', 'libx264', '-preset', 'slow', '-crf', '10', '-pix_fmt', 'yuv420p',
       '-movflags', '+faststart', OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
proc.stdin.close()
proc.wait()
if proc.returncode:
    raise SystemExit('ffmpeg failed')

if __name__ == '__main__':
    exact = np.array_equal(frames[0], first)
    print(f'wrote {OUT}: {len(frames)} frames, first frame exact match: {exact}')
