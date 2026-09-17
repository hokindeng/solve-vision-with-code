from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = (original[:, :, 1] > 150) & (original[:, :, 0] < 60) & (original[:, :, 2] < 60)
    red = (original[:, :, 0] > 200) & (original[:, :, 1] < 70) & (original[:, :, 2] < 70)
    gy, gx = np.where(green)
    ry, rx = np.where(red)
    # Include the complete border and its enclosed polygon, with a white margin.
    x0, x1 = gx.min() - 2, gx.max() + 3
    y0, y1 = gy.min() - 2, gy.max() + 3
    sprite = original[y0:y1, x0:x1].copy()
    background = original.copy()
    background[y0:y1, x0:x1] = 255
    distance = int(round((rx.min() + rx.max() - gx.min() - gx.max()) / 2))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(60):
        t = i / 59
        progress = t * t * (3 - 2 * t)
        dx = round(distance * progress)
        frame = background.copy()
        frame[y0:y1, x0 + dx:x1 + dx] = sprite
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    error = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
