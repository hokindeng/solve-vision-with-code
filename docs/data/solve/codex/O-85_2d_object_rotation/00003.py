from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The triangle is the sole foreground element below the title.
    mask = np.any(original < 255, axis=2)
    mask[:200] = False
    ys, xs = np.where(mask)
    background = original.copy()
    background[mask] = 255
    layer = np.full_like(original, 255)
    layer[mask] = original[mask]
    # Centroid of the triangle's three vertices (including its original outline).
    center = (511.0, (415.0 + 591.0 + 591.0) / 3.0)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '10',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(17):
        if i == 0:
            frame = original
        else:
            transform = cv2.getRotationMatrix2D(center, -180.0 * i / 16.0, 1.0)
            rotated = cv2.warpAffine(layer, transform, (1024, 1024),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            frame = np.minimum(background, rotated)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    err = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(err.decode())

if __name__ == '__main__':
    main()
