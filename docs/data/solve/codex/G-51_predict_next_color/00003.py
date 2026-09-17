from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Select only the white interior enclosed by the final circle.
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    interior = labels == labels[512, 912]
    target = original[512, 112].astype(float)
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(64):
        frame = original.copy()
        t = i / 63
        progress = t * t * (3 - 2 * t)
        frame[interior] = np.rint(255 + (target - 255) * progress).astype(np.uint8)
        assert np.array_equal(frame[~interior], original[~interior])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
