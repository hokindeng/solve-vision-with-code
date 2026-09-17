from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    color = np.array([209, 209, 209], dtype=np.uint8)
    ys, xs = np.where(np.all(original == color, axis=2))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = y1 - y0
    background = original.copy()
    background[y0:y1, x0:x1] = 255
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '12',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(58):
        top = round(y0 + (1024 - y0) * i / 57)
        frame = background.copy()
        if top < 1024:
            frame[top:min(top + height, 1024), x0:x1] = color
        if i == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    error = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
