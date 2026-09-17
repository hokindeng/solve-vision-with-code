from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(original == (0, 128, 128), axis=2).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    background = original.copy()
    blocks = []
    for x, y, w, h, area in stats[1:]:
        # Preserve the original two-pixel black outline and teal interior.
        x, y, w, h = int(x)-2, int(y)-2, int(w)+4, int(h)+4
        blocks.append((x, y, original[y:y+h, x:x+w].copy()))
        background[y:y+h, x:x+w] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-crf', '10', '-preset', 'slow', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(35):
        # Two coordinated one-cell steps, with gentle stops at cell centers.
        step = min(i // 17, 1)
        t = (i - step * 17) / 17
        eased = t*t*(3-2*t)
        shift = round(128 * (step + eased))
        frame = background.copy()
        for x, y, block in blocks:
            h, w = block.shape[:2]
            frame[y:y+h, x-shift:x-shift+w] = block
        if i == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
