from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(foreground, 8)
    # The filled hexagon and its surrounding green outline move together.
    hexagon_label = labels[374, 308]
    mask = (labels == hexagon_label) | np.all(original == (0, 200, 0), axis=2)
    ys, xs = np.nonzero(mask)
    pixels = original[ys, xs].copy()
    background = original.copy()
    background[mask] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(60):
        t = index / 59
        progress = t * t * (3 - 2 * t)
        shift = int(round(-172 * progress))
        frame = background.copy()
        frame[ys, xs + shift] = pixels
        if index == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
