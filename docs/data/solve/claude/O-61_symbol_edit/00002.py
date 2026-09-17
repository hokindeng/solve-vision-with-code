#!/usr/bin/env python3
"""Generate the 'insert hollow diamonds' video from first_frame.png.

Sequence layout (measured from first_frame.png): 9 slots, each a 1px grey
frame at x0 = 44 + 105*i .. x0+96, y = 464..560, interior 95x95.
Initial: 1 triangle, 2 square, 3 filled diamond, 4 circle, 5 hollow diamond.
Action: insert hollow diamond at 3 (shift 3..5 right), at 6 (shift 6 right),
and at 8 (empty slot).  Final: T S HD FD C HD HD HD _.
"""
import subprocess
import numpy as np
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 76
SLOT_X0 = [44 + 105 * i for i in range(9)]   # left border column of slot i
SLOT_Y0 = 464
SLOT_W = 97                                    # including 1px border each side
INNER = 95

base = np.array(Image.open('/app/first_frame.png').convert('RGB'))


def interior_box(i):
    x0 = SLOT_X0[i] + 1
    y0 = SLOT_Y0 + 1
    return x0, y0, x0 + INNER, y0 + INNER


def grab(i):
    x0, y0, x1, y1 = interior_box(i)
    return base[y0:y1, x0:x1].copy()


# Sprites (interior patches) of the initially filled slots
sprites = {i: grab(i) for i in range(5)}
hd = sprites[4].copy()                          # hollow diamond sprite (exact pixels)

# Mask of all slot interiors (sprites are only visible inside them)
inner_mask = np.zeros((H, W), bool)
for i in range(9):
    x0, y0, x1, y1 = interior_box(i)
    inner_mask[y0:y1, x0:x1] = True

# Background with slots 3..9 (indices 2..8) cleared to white
cleared = base.copy()
for i in range(2, 9):
    x0, y0, x1, y1 = interior_box(i)
    cleared[y0:y1, x0:x1] = 255


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def slot_pos(p):
    """Interior top-left for fractional slot index p."""
    return SLOT_X0[0] + 1 + 105.0 * p, SLOT_Y0 + 1


def paste_sprite(canvas, sprite, x, y):
    """Paste sprite (non-white pixels only) at float position, nearest px."""
    xi, yi = int(round(x)), int(round(y))
    h, w = sprite.shape[:2]
    region = canvas[yi:yi + h, xi:xi + w]
    m = sprite.sum(2) < 765
    region[m] = sprite[m]


def grow_sprite(sprite, s):
    """Sprite scaled by factor s about its centre, on white."""
    if s >= 0.999:
        return sprite
    h, w = sprite.shape[:2]
    m = (sprite.sum(2) < 765).astype(np.uint8) * 255
    rgba = np.dstack([sprite, m])
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = Image.fromarray(rgba, 'RGBA').resize((nw, nh), Image.LANCZOS)
    out = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    out.paste(small, ((w - nw) // 2, (h - nh) // 2))
    arr = np.array(out)
    alpha = arr[..., 3:4].astype(float) / 255.0
    rgb = arr[..., :3].astype(float) * alpha + 255.0 * (1 - alpha)
    return rgb.round().astype(np.uint8)


# ---- Timeline (frame indices) ----
# Step 1: shift slots 3,4,5 -> 4,5,6 then insert HD at 3
S1_MOVE = (4, 18)
S1_INS = (18, 30)
# Step 2: shift slot 6 (orig HD) -> 7 then insert HD at 6
S2_MOVE = (32, 44)
S2_INS = (44, 54)
# Step 3: insert HD at 8
S3_INS = (58, 70)


def prog(f, span):
    a, b = span
    return ease((f - a) / float(b - a))


def render(f):
    canvas = cleared.copy()
    layer = np.full((H, W, 3), 255, np.uint8)

    # existing movable sprites: (sprite, start_slot, moves list)
    p_fd = 2 + prog(f, S1_MOVE)             # filled diamond 3 -> 4
    p_ci = 3 + prog(f, S1_MOVE)             # circle 4 -> 5
    p_hd = 4 + prog(f, S1_MOVE) + prog(f, S2_MOVE)   # hollow diamond 5 -> 6 -> 7

    for spr, p in ((sprites[2], p_fd), (sprites[3], p_ci), (sprites[4], p_hd)):
        x, y = slot_pos(p)
        paste_sprite(layer, spr, x, y)

    # inserted hollow diamonds (grow in)
    for slot, span in ((2, S1_INS), (5, S2_INS), (7, S3_INS)):
        s = prog(f, span)
        if s <= 0:
            continue
        x, y = slot_pos(slot)
        paste_sprite(layer, grow_sprite(hd, s), x, y)

    m = inner_mask & (layer.sum(2) < 765)
    canvas[m] = layer[m]
    return canvas


def main():
    frames = [render(f) for f in range(N_FRAMES)]
    assert np.array_equal(frames[0], base), 'first frame must match first_frame.png'
    Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'medium',
           '-r', str(FPS), '/app/output/video.mp4']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0


if __name__ == '__main__':
    main()
