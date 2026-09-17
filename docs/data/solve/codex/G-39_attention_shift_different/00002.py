from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = np.array([70, 140, 70], dtype=np.uint8)
    mask = np.all(source == green, axis=2)
    ys, xs = np.where(mask)
    background = source.copy()
    background[mask] = 255
    # Translate the original box, preserving its dimensions and raster outline.
    # Center it on the right object's bounding rectangle.
    right_object = np.all(source == [146, 231, 140], axis=2)
    ry, rx = np.where(right_object)
    dx = round((rx.min() + rx.max() - xs.min() - xs.max()) / 2)
    dy = round((ry.min() + ry.max() - ys.min() - ys.max()) / 2)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', '1024x1024',
               '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(25):
        t = i / 24
        eased = t * t * (3 - 2 * t)
        frame = background.copy()
        frame[ys + round(dy * eased), xs + round(dx * eased)] = green
        if i == 0:
            assert np.array_equal(frame, source)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
