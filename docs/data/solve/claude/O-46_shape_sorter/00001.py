#!/usr/bin/env python3
"""Shape sorter: slide each colored card from the left staging area into its matching outline."""
import subprocess
import numpy as np
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 78
BG = np.array([248, 250, 252], dtype=np.uint8)

first = np.array(Image.open('/app/first_frame.png').convert('RGB'))

# (card color, starting bbox x0,x1,y0,y1 inclusive, target center)
CARDS = [
    ((251, 146, 60), (151, 241, 302, 392), (757.5, 293.5)),   # orange square -> square outline
    ((244, 114, 182), (290, 380, 294, 384), (757.5, 511.5)),  # pink diamond -> diamond outline
    ((250, 204, 21), (151, 241, 633, 723), (757.0, 730.0)),   # yellow circle -> circle outline
]

# Extract sprites and build the static background (scene with the cards removed).
base = first.copy()
sprites = []
for color, (x0, x1, y0, y1), tgt in CARDS:
    region = first[y0:y1 + 1, x0:x1 + 1]
    mask = (region == np.array(color, dtype=np.uint8)).all(-1)
    sprites.append((region.copy(), mask))
    base[y0:y1 + 1, x0:x1 + 1][mask] = BG
    h, w = mask.shape
    start = np.array([x0, y0], dtype=float)
    end = np.array([tgt[0] - (w - 1) / 2.0, tgt[1] - (h - 1) / 2.0])
    sprites[-1] = (region.copy(), mask, start, np.round(end))


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


# Schedule: frame 0 static; three moves of equal length; final frame holds result.
move_len = (N_FRAMES - 2) // 3          # 25 frames each
segments = []
f = 1
for i in range(3):
    segments.append((f, f + move_len - 1))
    f += move_len
# Any leftover frames hold at the end.


def card_pos(i, frame):
    _, _, start, end = sprites[i]
    s0, s1 = segments[i]
    if frame < s0:
        return start
    if frame > s1:
        return end
    t = (frame - s0 + 1) / (s1 - s0 + 1)
    return np.round(start + (end - start) * ease(t))


def blit(img, sprite, mask, x, y):
    x, y = int(x), int(y)
    h, w = mask.shape
    img[y:y + h, x:x + w][mask] = sprite[mask]


def render(frame):
    img = base.copy()
    # Draw stationary cards first, the moving card last so it stays on top.
    order = sorted(range(3), key=lambda i: segments[i][0] <= frame <= segments[i][1])
    for i in order:
        sprite, mask, _, _ = sprites[i]
        x, y = card_pos(i, frame)
        blit(img, sprite, mask, x, y)
    return img


def main():
    import os
    os.makedirs('/app/output', exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow',
           '/app/output/video.mp4']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        img = render(f)
        if f == 0:
            assert (img == first).all(), 'first frame mismatch'
        p.stdin.write(img.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(render(N_FRAMES - 1)).save('/app/output/last_frame.png')


if __name__ == '__main__':
    main()
