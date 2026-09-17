from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    # This point is strictly inside all three circles. Their original strokes
    # enclose exactly the white component to be colored.
    region_id = labels[480, 500]
    mask = labels == region_id
    assert 1000 < mask.sum() < 50000
    yy, xx = np.nonzero(mask)
    top, bottom = yy.min(), yy.max()
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
           'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
           '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rows = np.arange(original.shape[0])[:, None]
    for i in range(60):
        frame = original.copy()
        if i:
            progress = i / 59
            # A continuous downward sweep paints only white overlap pixels.
            edge = top + progress * (bottom - top + 1)
            coverage = np.clip(edge - rows, 0, 1)
            a = coverage[yy, 0]
            frame[yy, xx] = np.rint(
                (1 - a[:, None]) * 255 + a[:, None] * np.array([255, 0, 0])
            ).astype(np.uint8)
        assert np.array_equal(frame[~mask], original[~mask])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
