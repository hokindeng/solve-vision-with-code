#!/usr/bin/env python3
"""Slide the three left-hand objects horizontally into their dashed targets."""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
W = H = 1024
FPS = 16
N_FRAMES = 30
# Horizontal displacement (px) per object, keyed by the object's bbox left x.
# Measured from first_frame.png: big circle, small circle, triangle.
DX_BY_LEFT = {60: 637, 141: 450, 260: 432}


def main():
    first = np.array(Image.open(SRC).convert('RGB'))
    diff = np.abs(first.astype(int) - 255).sum(2)
    strong = diff > 30
    faint = diff > 0
    left = strong.copy()
    left[:, 500:] = False  # dashed targets live on the right half
    n, lab = cv2.connectedComponents(left.astype(np.uint8))

    sprites = []
    bg = first.copy()
    for k in range(1, n):
        comp = lab == k
        # pull in anti-aliased fringe pixels around the component
        grown = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        mask = grown & faint
        ys, xs = np.where(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        dx = DX_BY_LEFT[min(DX_BY_LEFT, key=lambda l: abs(l - x0))]
        sprites.append(dict(rgb=first[y0:y1, x0:x1].copy(),
                            a=mask[y0:y1, x0:x1].copy(), x=x0, y=y0, dx=dx))
        bg[mask] = 255  # erase object; background elsewhere untouched

    def render(t):
        s = t * t * (3 - 2 * t)  # smoothstep ease in/out
        frame = bg.copy()
        for sp in sprites:
            x = sp['x'] + int(round(sp['dx'] * s))
            h, w = sp['a'].shape
            region = frame[sp['y']:sp['y'] + h, x:x + w]
            region[sp['a']] = sp['rgb'][sp['a']]
        return frame

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'slow', OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        frame = first if i == 0 else render(t)
        p.stdin.write(np.ascontiguousarray(frame).tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(render(1.0)).save('/app/output/last_frame.png')


if __name__ == '__main__':
    main()
