from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    orange = np.array([217, 119, 6], dtype=np.uint8)
    # Exact interior bounds; the grid is never painted over.
    cells = [(718, 309, 818, 408),
             (514, 514, 613, 613),
             (616, 514, 715, 613),
             (718, 514, 818, 613),
             (514, 718, 613, 818)]
    allowed = np.zeros(base.shape[:2], dtype=bool)
    for x0, y0, x1, y1 in cells:
        allowed[y0:y1, x0:x1] = True
    assert np.all(base[allowed] == 255)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')], stdin=subprocess.PIPE)
    for i in range(35):
        frame = base.copy()
        progress = i / 34 * len(cells)
        for j, (x0, y0, x1, y1) in enumerate(cells):
            fraction = np.clip(progress - j, 0, 1)
            width = int(round((x1 - x0) * fraction))
            frame[y0:y1, x0:x0 + width] = orange
        assert np.array_equal(frame[~allowed], base[~allowed])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
