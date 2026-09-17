from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(original == (209, 209, 209), axis=2)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rectangle = original[y0:y1, x0:x1].copy()
    background = original.copy()
    background[y0:y1, x0:x1] = 255
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '10', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    for i in range(58):
        top = round(y0 + (1024 - y0) * i / 57)
        frame = background.copy()
        bottom = min(1024, top + y1 - y0)
        if top < 1024:
            frame[top:bottom, x0:x1] = rectangle[:bottom-top]
        if i == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
