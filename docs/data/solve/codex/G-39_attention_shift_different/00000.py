from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Isolate the green outline, including any edge pixels, from the fixed art.
    r, g, b = original.astype(np.int16).transpose(2, 0, 1)
    outline = (g > r) & (g > b)
    ys, xs = np.nonzero(outline)
    colors = original[ys, xs].copy()
    background = original.copy()
    background[outline] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(25):
        t = i / 24
        # Smooth acceleration/deceleration, with movement spanning the full clip.
        shift = round(503 * (t * t * (3 - 2 * t)))
        frame = background.copy()
        frame[ys, xs + shift] = colors
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
