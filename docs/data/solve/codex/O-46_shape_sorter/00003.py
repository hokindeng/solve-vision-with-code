from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = np.array([248, 250, 252], dtype=np.uint8)
    colors = [(250, 204, 21), (96, 165, 250), (34, 211, 238), (251, 146, 60)]
    # Align each card's center with the corresponding outline's center.
    offsets = [(491.5, -6.5), (491.5, -8.5), (488.5, 5.5), (490, 5)]
    masks = [np.all(original == color, axis=2).astype(np.float32) for color in colors]
    base = original.copy()
    for mask in masks:
        base[mask > 0] = background
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '10', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(80):
        canvas = base.copy()
        for index, (mask, color, (dx, dy)) in enumerate(zip(masks, colors, offsets)):
            t = np.clip((frame - (2 + 18 * index)) / 18.0, 0, 1)
            t = t * t * (3 - 2 * t)
            if t == 0:
                moved = mask
            else:
                moved = cv2.warpAffine(mask, np.float32([[1, 0, dx*t], [0, 1, dy*t]]),
                                       (1024, 1024), flags=cv2.INTER_LINEAR)
            active = moved > 0
            alpha = moved[active, None]
            canvas[active] = np.rint(canvas[active] * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
        if frame == 0:
            assert np.array_equal(canvas, original)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
