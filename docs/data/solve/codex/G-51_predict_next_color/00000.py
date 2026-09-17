from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Select just the enclosed white interior, preserving the original border.
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    interior = labels == labels[512, 912]
    color = original[512, 312].astype(float)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(64):
        frame = original.copy()
        t = np.clip((i - 4) / 55.0, 0, 1)
        amount = t * t * (3 - 2 * t)
        frame[interior] = np.rint(255 * (1 - amount) + color * amount).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
