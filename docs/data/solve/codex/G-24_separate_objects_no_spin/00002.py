from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Each source includes its original antialiased edge and unchanged color.
    specs = [(59, 326, 242, 509, 637),
             (140, 551, 243, 654, 450),
             (259, 506, 425, 674, 430)]
    background = original.copy()
    sprites = []
    for x0, y0, x1, y1, distance in specs:
        patch = original[y0:y1, x0:x1].copy()
        mask = np.any(patch != 255, axis=2)
        sprites.append((x0, y0, patch, mask, distance))
        background[y0:y1, x0:x1][mask] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '16',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(30):
        t = i / 29
        progress = t * t * (3 - 2 * t)
        frame = background.copy()
        for x, y, patch, mask, distance in sprites:
            dest_x = x + round(distance * progress)
            h, w = mask.shape
            area = frame[y:y+h, dest_x:dest_x+w]
            area[mask] = patch[mask]
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
