from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask_color = np.array([209, 209, 209], dtype=np.uint8)
    ys, xs = np.where(np.all(original == mask_color, axis=2))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = y1 - y0
    background = original.copy()
    background[y0:y1, x0:x1] = 255
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
               '-c:v', 'libx264', '-crf', '0', '-preset', 'slow',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(58):
        frame = background.copy()
        top = round(y0 + (1024 - y0) * index / 57)
        bottom = min(1024, top + height)
        if top < 1024:
            frame[top:bottom, x0:x1] = mask_color
        if index == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
