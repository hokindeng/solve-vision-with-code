from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mixed = np.rint(np.array([120, 25, 0]) * np.array([149, 163, 214]) / 255).astype(np.uint8)
    # The inner edges of the original six-pixel black border.
    x0, x1, y0, y1 = 395, 630, 396, 629
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(44):
        frame = original.copy()
        # A steadily rising fill keeps the action distributed across the clip.
        filled = (y1-y0) * i / 43
        rows = np.arange(y0, y1)
        coverage = np.clip(filled - (y1 - 1 - rows), 0, 1)[:, None, None]
        region = original[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = np.rint(region * (1-coverage) + mixed * coverage).astype(np.uint8)
        assert np.array_equal(frame[:y0], original[:y0])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
