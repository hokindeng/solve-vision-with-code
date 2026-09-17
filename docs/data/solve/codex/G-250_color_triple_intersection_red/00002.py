from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The white component enclosed by all three intact outlines is the overlap.
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    region_id = labels[500, 480]
    region = labels == region_id
    x, y, width, height, area = stats[region_id]
    assert 19000 < area < 22000
    rows = np.arange(original.shape[0])[:, None]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(60):
        frame = original.copy()
        # A steady downward fill occupies the entire duration; frame zero is exact.
        progress = frame_index / 59
        fill = region & (rows < y + int(np.ceil(height * progress)))
        frame[fill] = (255, 0, 0)
        assert np.array_equal(frame[~region], original[~region])
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
