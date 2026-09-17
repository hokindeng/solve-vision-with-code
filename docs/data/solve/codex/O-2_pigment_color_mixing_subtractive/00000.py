from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Select only the white interior enclosed by the existing black border.
    white = np.all(base == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    mask = labels == labels[512, 512]
    pigment1 = np.array([204, 207, 223], dtype=np.float64)
    pigment2 = np.array([39, 232, 220], dtype=np.float64)
    mixed = np.rint(pigment1 * pigment2 / 255).astype(np.uint8)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
               'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(44):
        t = index / 43
        amount = t * t * (3 - 2 * t)
        frame = base.copy()
        frame[mask] = np.rint(255 * (1 - amount) + mixed * amount).astype(np.uint8)
        assert np.array_equal(frame[~mask], base[~mask])
        if index == 0:
            assert np.array_equal(frame, base)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    print(f'Mixed RGB: {tuple(mixed.tolist())}; wrote {out / "video.mp4"}')

if __name__ == '__main__':
    main()
