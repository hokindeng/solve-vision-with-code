from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    pigment_left = np.array([204, 171, 33], dtype=np.float64)
    pigment_right = np.array([136, 88, 158], dtype=np.float64)
    mixed = np.rint(pigment_left * pigment_right / 255).astype(np.uint8)
    # The open interior of the black square; its border is preserved.
    interior = np.zeros(original.shape[:2], dtype=bool)
    interior[396:629, 395:630] = True
    assert np.all(original[interior] == 255)
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(44):
        t = i / 43
        progress = t * t * (3 - 2 * t)
        frame = original.copy()
        frame[interior] = np.rint(255 * (1 - progress) + mixed * progress).astype(np.uint8)
        assert np.array_equal(frame[~interior], original[~interior])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')
    print('Subtractive mix:', tuple(int(v) for v in mixed))

if __name__ == '__main__':
    main()
