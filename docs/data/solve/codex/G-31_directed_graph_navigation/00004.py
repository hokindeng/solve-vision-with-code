from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Preserve the exact raster sprite, including its dark blue outline.
    mask = (original[:, :, 2] > original[:, :, 0]) & (original[:, :, 2] > original[:, :, 1])
    ys, xs = np.where(mask)
    sprite = original[ys, xs].copy()
    background = original.copy()
    background[ys, xs] = (0, 128, 0)
    start = np.array([724., 349.])
    middle = np.array([587., 643.])
    end = np.array([277., 793.])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
           '-video_size', '1024x1024', '-framerate', '16', '-i', '-', '-an',
           '-c:v', 'libx264', '-crf', '0', '-preset', 'slow', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(30):
        if i <= 1:
            frame = original.copy()
        else:
            if i <= 14:
                position = start + (middle - start) * ((i - 1) / 13)
            elif i == 15:
                position = middle
            else:
                position = middle + (end - middle) * min(1., (i - 15) / 13)
            dx, dy = np.rint(position - start).astype(int)
            frame = background.copy()
            frame[ys + dy, xs + dx] = sprite
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
