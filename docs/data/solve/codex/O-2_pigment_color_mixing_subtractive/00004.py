"""Animate the subtractive pigment mixture within the original mixing zone."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Find only the enclosed white region, preserving every border pixel.
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    mask = labels == labels[512, 512]
    c1 = np.array([25, 54, 147], dtype=np.float64)
    c2 = np.array([163, 122, 58], dtype=np.float64)
    mixed = np.rint(c1 * c2 / 255).astype(np.uint8)
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output)
    ], stdin=subprocess.PIPE)
    for i in range(44):
        t = i / 43
        amount = t * t * (3 - 2 * t)
        frame = original.copy()
        frame[mask] = np.rint(255 * (1 - amount) + mixed * amount).astype(np.uint8)
        assert np.array_equal(frame[~mask], original[~mask])
        if i == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')
    print(f'Mixed RGB: {tuple(mixed.tolist())}; wrote {output}')

if __name__ == '__main__':
    main()
