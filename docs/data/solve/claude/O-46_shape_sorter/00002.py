#!/usr/bin/env python3
"""Shape sorter: slide each colored card from the left staging area into its
matching outline on the right. Renders /app/output/video.mp4 (1024x1024, 16 fps, 77 frames)."""
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS = 16
N_FRAMES = 77

im = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = im.shape
BG = im[5, 5].copy()
OUTLINE = np.array([100, 116, 139])

# Card colors in the order they must be moved (yellow circle, red triangle,
# orange star, cyan square, pink hexagon).
CARD_COLORS = [
    (250, 204, 21),   # yellow circle
    (248, 113, 113),  # red triangle
    (251, 146, 60),   # orange star
    (34, 211, 238),   # cyan square
    (244, 114, 182),  # pink hexagon
]


def bbox_center(mask):
    ys, xs = np.where(mask)
    return (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0


# Extract card sprites (mask + pixels) and starting centers.
cards = []
for col in CARD_COLORS:
    mask = (im == np.array(col)).all(2)
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cards.append({
        'mask': mask[y0:y1, x0:x1],
        'pix': im[y0:y1, x0:x1].copy(),
        'start': np.array([x0, y0], dtype=float),
        'size': np.array([x1 - x0, y1 - y0]),
    })

# Outline components -> match to cards by shape (bbox size + fill fraction is
# ambiguous, so match by vertical row and left/right column position which is
# the layout of the puzzle: same grid order on both sides).
omask = (im == OUTLINE).all(2)
lab, n = ndimage.label(omask)
outlines = []
for i in range(1, n + 1):
    m = lab == i
    ys, xs = np.where(m)
    outlines.append({'center': np.array([(xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0]),
                     'bbox': (xs.min(), xs.max(), ys.min(), ys.max())})

# Match each card to the outline whose shape fits: for each outline, compute the
# filled interior shape and compare to card masks by normalized IoU.
def filled_interior(bbox):
    x0, x1, y0, y1 = bbox
    sub = omask[y0:y1 + 1, x0:x1 + 1]
    filled = ndimage.binary_fill_holes(sub)
    return filled

def iou(a, b):
    # resize b to a's shape via nearest
    from PIL import Image as _I
    bi = np.array(_I.fromarray(b.astype(np.uint8) * 255).resize((a.shape[1], a.shape[0]), _I.NEAREST)) > 127
    return (a & bi).sum() / max(1, (a | bi).sum())

targets = []
used = set()
for c in cards:
    best, bi = -1, None
    for j, o in enumerate(outlines):
        if j in used:
            continue
        s = iou(c['mask'], filled_interior(o['bbox']))
        if s > best:
            best, bi = s, j
    used.add(bi)
    targets.append(outlines[bi]['center'])

# Base frame: the first frame with all cards erased.
base = im.copy()
for col in CARD_COLORS:
    base[(im == np.array(col)).all(2)] = BG


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


# Timeline: frame 0 = first frame. Then 5 moves of 15 frames each (1..75),
# frame 76 holds the finished result.
MOVE_LEN = 15
positions = []  # per frame, per card top-left (x, y)
for f in range(N_FRAMES):
    pos = []
    for k, c in enumerate(cards):
        start_tl = c['start']
        end_tl = targets[k] - (c['size'] - 1) / 2.0
        m0 = 1 + k * MOVE_LEN
        m1 = m0 + MOVE_LEN - 1  # last frame of the move (card arrives)
        if f < m0:
            t = 0.0
        elif f >= m1:
            t = 1.0
        else:
            t = ease((f - m0 + 1) / float(MOVE_LEN))
        pos.append(np.round(start_tl + (end_tl - start_tl) * t).astype(int))
    positions.append(pos)


def render(pos):
    fr = base.copy()
    for c, (x, y) in zip(cards, pos):
        h, w = c['mask'].shape
        region = fr[y:y + h, x:x + w]
        region[c['mask']] = c['pix'][c['mask']]
    return fr


frames = [render(p) for p in positions]
assert np.array_equal(frames[0], im), 'first frame must be unchanged'

proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'medium',
    '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
for fr in frames:
    proc.stdin.write(fr.astype(np.uint8).tobytes())
proc.stdin.close()
proc.wait()
print('wrote', OUT, len(frames), 'frames')
