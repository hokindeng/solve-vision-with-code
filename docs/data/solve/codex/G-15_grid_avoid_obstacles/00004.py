from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Preserve the original two-color agent, including its original outline.
    mask = ((original == (255, 200, 0)).all(axis=2) |
            (original == (200, 150, 0)).all(axis=2))
    ys, xs = np.where(mask)
    sprite = original[ys, xs].copy()
    background = original.copy()
    background[ys, xs] = (0, 100, 255)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Cells (row 7, columns 2 through 5) form the shortest unobstructed path.
    # Each of the three 93-pixel moves takes seven frame intervals.
    for i in range(22):
        frame = background.copy()
        dx = round(279 * i / 21)
        frame[ys, xs + dx] = sprite
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
