from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Select only blue pixels, including any antialiased boundary pixels.
    r, g, b = original.astype(np.int16).transpose(2, 0, 1)
    mask = (b > r) & (b > g) & (r == g)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(96):
        t = index / 95
        fade = t * t * (3 - 2 * t)
        frame = original.copy()
        frame[mask] = np.rint(original[mask].astype(float) * (1 - fade) + 255 * fade).astype(np.uint8)
        assert np.array_equal(frame[~mask], original[~mask])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
