from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    count, labels = cv2.connectedComponents(foreground, connectivity=8)
    objects = []
    totals = np.zeros(3, dtype=int)
    for label in range(1, count):
        pixels = original[labels == label].astype(np.int16)
        colored = pixels.max(axis=1) - pixels.min(axis=1) > 100
        if not colored.any():
            continue
        color = int(pixels[colored].mean(axis=0).argmax())
        totals[color] += 1
        objects.append((label, color))
    majority = int(totals.argmax())
    remove = np.isin(labels, [label for label, color in objects if color != majority])
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(40):
        t = index / 39
        alpha = t * t * (3 - 2 * t)
        frame = original.copy()
        frame[remove] = np.rint(original[remove].astype(float) * (1-alpha) + 255 * alpha).astype(np.uint8)
        assert np.array_equal(frame[~remove], original[~remove])
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')
    print('Object counts (red, green, blue):', totals.tolist())

if __name__ == '__main__':
    main()
