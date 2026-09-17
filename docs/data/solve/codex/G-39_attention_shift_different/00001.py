from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    src = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Extract the existing box so its color, thickness, and corner pixels survive.
    box = np.all(src == [70, 140, 70], axis=2)
    ys, xs = np.nonzero(box)
    background = src.copy()
    background[box] = 255
    # The hexagon is centered at x=733; retain the original box dimensions.
    displacement = 733 - (xs.min() + xs.max()) / 2
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(25):
        t = i / 24
        progress = t * t * (3 - 2 * t)
        dx = round(displacement * progress)
        frame = background.copy()
        frame[ys, xs + dx] = src[ys, xs]
        if i == 0:
            assert np.array_equal(frame, src)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
