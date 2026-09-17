from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background_color = original[0, 0].copy()
    # Each crop contains one entire face. Offsets align the face with the
    # corresponding outline's head, retaining the original size and artwork.
    animals = [
        ((78, 177, 223, 321), (507, -30)),  # panda: upper round-eared outline
        ((303, 96, 477, 263), (268, 323)),  # bear: larger round-eared outline
        ((196, 415, 365, 583), (377, 302)), # lion: petal-shaped mane
        ((95, 727, 217, 861), (693, -580)), # cat: plain pointed ears
        ((322, 745, 446, 870), (465, -305)),# tiger: pointed ears and stripes
    ]
    stationary = original.copy()
    sprites = []
    for (x0, y0, x1, y1), (dx, dy) in animals:
        crop = original[y0:y1, x0:x1].copy()
        mask = np.any(crop != background_color, axis=2)
        stationary[y0:y1, x0:x1][mask] = background_color
        sprites.append((x0, y0, crop, mask, dx, dy))

    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(64):
        t = i / 63.0
        frame = stationary.copy()
        for x0, y0, crop, mask, dx, dy in sprites:
            x = x0 + round(dx * t)
            y = y0 + round(dy * t)
            h, w = mask.shape
            frame[y:y+h, x:x+w][mask] = crop[mask]
        if i == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
