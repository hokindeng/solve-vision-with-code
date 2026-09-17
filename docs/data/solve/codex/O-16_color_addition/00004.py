from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    h, w = original.shape[:2]
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    _, labels, _, centers = cv2.connectedComponentsWithStats(foreground)
    masks = [(labels == k).astype(np.uint8) for k in (1, 2)]
    sprites = [original * mask[..., None] for mask in masks]
    midpoint = (centers[1] + centers[2]) / 2
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    writer = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(80):
        if frame == 0:
            result = original
        else:
            t = frame / 79
            coverage = np.zeros((h, w), dtype=np.uint8)
            light = np.zeros((h, w, 3), dtype=np.uint16)
            for k in range(2):
                offset = (midpoint - centers[k + 1]) * t
                transform = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
                mask = cv2.warpAffine(masks[k], transform, (w, h), flags=cv2.INTER_NEAREST)
                color = cv2.warpAffine(sprites[k], transform, (w, h), flags=cv2.INTER_NEAREST)
                coverage |= mask
                light += color.astype(np.uint16)
            result = np.full_like(original, 255)
            result[coverage != 0] = np.minimum(light[coverage != 0], 255).astype(np.uint8)
        writer.stdin.write(result.tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
