from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Flood fill the enclosed white interior, leaving every border pixel intact.
    white = np.all(original == 255, axis=2)
    interior = np.zeros(white.shape, dtype=bool)
    pending = [(512, 512)]
    while pending:
        x, y = pending.pop()
        if not (0 <= x < white.shape[1] and 0 <= y < white.shape[0]):
            continue
        if interior[y, x] or not white[y, x]:
            continue
        interior[y, x] = True
        pending.extend(((x-1,y), (x+1,y), (x,y-1), (x,y+1)))
    mix = np.rint(np.array([136,158,164]) * np.array([68,180,242]) / 255).astype(np.uint8)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(44):
        t = index / 43
        amount = t*t*(3-2*t)
        frame = original.copy()
        frame[interior] = np.rint(255*(1-amount) + mix.astype(float)*amount).astype(np.uint8)
        assert np.array_equal(frame[~interior], original[~interior])
        if index == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
