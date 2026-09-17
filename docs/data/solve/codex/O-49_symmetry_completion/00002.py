from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    blue = np.array([37, 99, 235], dtype=np.uint8)
    # Only the white interiors of the six missing cells may change.
    cells = [(3, 5), (5, 5), (5, 7), (7, 5), (7, 6), (7, 7)]
    masks = []
    yy, xx = np.indices(original.shape[:2])
    white = np.all(original == 255, axis=2)
    for row, col in cells:
        mask = white & (xx >= 204 + col * 76.8) & (xx < 204 + (col + 1) * 76.8) & (yy >= 204 + row * 76.8) & (yy < 204 + (row + 1) * 76.8)
        xs = xx[mask]
        masks.append((mask, int(xs.min()), int(xs.max()) + 1))
    allowed = np.logical_or.reduce([m[0] for m in masks])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_index in range(35):
        frame = original.copy()
        progress = frame_index * len(cells) / 34
        for i, (mask, left, right) in enumerate(masks):
            fraction = np.clip(progress - i, 0, 1)
            edge = left + int(round((right - left) * fraction))
            frame[mask & (xx < edge)] = blue
        assert np.array_equal(frame[~allowed], original[~allowed])
        if frame_index == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
