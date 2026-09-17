from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    circles = []
    background = source.copy()
    for color in ((230, 25, 75), (250, 190, 212)):
        mask = np.all(source == color, axis=2)
        y, x = np.where(mask)
        center = np.array([(x.min() + x.max()) / 2, (y.min() + y.max()) / 2])
        circles.append((y, x, color, np.array([512, 512]) - center))
        background[mask] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(40):
        t = i / 39
        progress = t * t * (3 - 2 * t)
        frame = background.copy()
        for y, x, color, displacement in circles:
            dx, dy = np.rint(displacement * progress).astype(int)
            frame[y + dy, x + dx] = color
        if i == 0:
            assert np.array_equal(frame, source)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
