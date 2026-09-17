from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    src = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Isolate the original attention outline, including its rounded corners.
    yy, xx = np.indices(src.shape[:2])
    region = (xx >= 97) & (xx <= 455) & (yy >= 333) & (yy <= 691)
    border = (xx < 110) | (xx > 442) | (yy < 347) | (yy > 677)
    mask = region & border & np.any(src != 255, axis=2)
    sy, sx = np.where(mask)
    colors = src[sy, sx].copy()
    background = src.copy()
    background[mask] = 255
    # Center the unchanged-size outline on the right-hand triangle's bounds.
    right = (xx > 600) & np.any(src != 255, axis=2)
    ry, rx = np.where(right)
    dx = (rx.min() + rx.max() - sx.min() - sx.max()) / 2
    dy = (ry.min() + ry.max() - sy.min() - sy.max()) / 2
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pixel_format', 'rgb24', '-video_size', '1024x1024', '-framerate', '16',
           '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'slow',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(25):
        t = i / 24
        t = t * t * (3 - 2 * t)
        frame = background.copy()
        frame[sy + round(dy * t), sx + round(dx * t)] = colors
        if i == 0:
            assert np.array_equal(frame, src)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
