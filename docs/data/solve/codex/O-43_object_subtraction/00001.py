from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    r, g, b = original.astype(np.int16).transpose(2, 0, 1)
    orange = (r > 220) & (g > 65) & (g < 190) & (b < 60)
    assert orange.any(), 'No orange object found'
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    height, width = original.shape[:2]
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(96):
            t = i / 95
            amount = t * t * (3 - 2 * t)
            frame = original.copy()
            frame[orange] = np.rint(original[orange].astype(float) * (1 - amount) + 255 * amount).astype(np.uint8)
            assert np.array_equal(frame[~orange], original[~orange])
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
