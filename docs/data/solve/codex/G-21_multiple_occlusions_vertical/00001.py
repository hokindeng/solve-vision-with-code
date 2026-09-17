"""Animate the original rectangular occluder without redrawing the scene."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask_color = np.array([209, 209, 209], dtype=np.uint8)
    ys, xs = np.where(np.all(source == mask_color, axis=2))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = y1 - y0
    background = source.copy()
    background[y0:y1, x0:x1] = 255
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(58):
        top = round(y0 + (1024 - y0) * index / 57)
        frame = background.copy()
        if top < 1024:
            frame[top:min(top + height, 1024), x0:x1] = mask_color
        if index == 0:
            assert np.array_equal(frame, source)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
