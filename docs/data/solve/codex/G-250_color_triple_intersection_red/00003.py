from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    # This seed lies inside all three circles. The original strokes bound
    # the connected white region exactly, so every outline is preserved.
    region = labels == labels[475, 500]
    ys, xs = np.where(region)
    assert 1000 < region.sum() < 100000
    assert xs.min() > 400 and xs.max() < 620
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
           'medium', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
           '+faststart', str(output / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_number in range(60):
        frame = original.copy()
        progress = frame_number / 59.0
        # A steady downward paint front spans the whole animation.
        front = ys.min() + progress * (ys.max() - ys.min() + 1)
        coverage = np.clip(front - ys, 0, 1)
        frame[ys, xs] = np.round(
            (1 - coverage[:, None]) * original[ys, xs] +
            coverage[:, None] * np.array([255, 0, 0])
        ).astype(np.uint8)
        assert np.array_equal(frame[~region], original[~region])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
