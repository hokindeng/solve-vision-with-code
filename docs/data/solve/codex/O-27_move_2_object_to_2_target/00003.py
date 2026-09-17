from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    # Extract only the two solid objects, retaining every original edge pixel.
    objects = []
    for (x0, y0, x1, y1), displacement in [
        ((257, 347, 400, 490), (273, 120)),
        ((838, 810, 974, 939), (-275, -16)),
    ]:
        patch = original[y0:y1, x0:x1].copy()
        mask = np.any(patch != (220, 220, 220), axis=2)
        background[y0:y1, x0:x1][mask] = (220, 220, 220)
        objects.append((x0, y0, patch, mask, displacement))

    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4'),
    ], stdin=subprocess.PIPE)
    for frame_index in range(35):
        t = frame_index / 34
        frame = background.copy()
        for x0, y0, patch, mask, (dx, dy) in objects:
            x = x0 + round(dx * t)
            y = y0 + round(dy * t)
            h, w = mask.shape
            frame[y:y+h, x:x+w][mask] = patch[mask]
        if frame_index == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
