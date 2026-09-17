from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    src = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = np.all(src == (0, 200, 0), axis=2)
    gy, gx = np.where(green)
    # Preserve the original raster of the entire bordered object, including its white gap.
    x0, x1 = int(gx.min()), int(gx.max()) + 1
    y0, y1 = int(gy.min()), int(gy.max()) + 1
    sprite = src[y0:y1, x0:x1].copy()
    base = src.copy()
    base[y0:y1, x0:x1] = 255
    ty, tx = np.where(np.all(src == (30, 142, 153), axis=2))
    displacement = int(round((tx.min() + tx.max()) / 2 - (gx.min() + gx.max()) / 2))
    (ROOT / 'output').mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(ROOT / 'output/video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(60):
        t = i / 59
        shift = round(displacement * (t * t * (3 - 2 * t)))
        frame = base.copy()
        frame[y0:y1, x0 + shift:x1 + shift] = sprite
        if i == 0:
            assert np.array_equal(frame, src)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    err = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(err.decode())

if __name__ == '__main__':
    main()
