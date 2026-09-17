from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    source = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The marked card is the second symbol. Its entire extent, including
    # the red deletion marker, lies within this rectangle.
    mask = np.zeros(source.shape[:2], dtype=bool)
    mask[446:579, 311:444] = True
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(46):
        # Hold briefly to identify the target, then erase it smoothly.
        t = float(np.clip((i - 5) / 35, 0, 1))
        t = t * t * (3 - 2 * t)
        frame = source.copy()
        frame[mask] = np.rint(source[mask].astype(float) * (1 - t) + 255 * t).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
