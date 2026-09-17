from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The existing strokes bound each Venn region. Select the enclosed white
    # component common to all three circles, leaving every stroke untouched.
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    mask = labels == labels[520, 490]
    yy, xx = np.nonzero(mask)
    assert 20000 < len(yy) < 22000
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')], stdin=subprocess.PIPE)
    for index in range(60):
        frame = original.copy()
        if index:
            progress = index / 59
            edge = yy.min() - 1 + progress * (yy.max() - yy.min() + 3)
            coverage = np.clip(edge - yy, 0, 1)
            frame[yy, xx, 1] = np.rint(255 * (1 - coverage)).astype(np.uint8)
            frame[yy, xx, 2] = np.rint(255 * (1 - coverage)).astype(np.uint8)
        assert np.array_equal(frame[~mask], original[~mask])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
