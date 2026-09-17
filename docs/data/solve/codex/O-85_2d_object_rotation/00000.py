from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(source != 255, axis=2).astype(np.uint8)
    foreground[:100] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, 8)
    objects = []
    background = source.copy()
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 1000:
            continue
        mask = labels == label
        layer = np.full_like(source, 255)
        layer[mask] = source[mask]
        background[mask] = 255
        objects.append((layer, tuple(centroids[label])))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'slow',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(17):
        if index == 0:
            frame = source
        else:
            frame = background.copy()
            angle = 176.0 * index / 16
            for layer, center in objects:
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(layer, matrix, (1024, 1024),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))
                frame = np.minimum(frame, rotated)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    error = encoder.stderr.read()
    if encoder.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
