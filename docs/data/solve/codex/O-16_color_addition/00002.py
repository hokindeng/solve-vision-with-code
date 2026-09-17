from pathlib import Path
import cv2
import numpy as np
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    colors = [np.array([97, 56, 74], dtype=np.float32), np.array([113, 186, 93], dtype=np.float32)]
    _, labels, stats, centers = cv2.connectedComponentsWithStats(np.uint8(np.any(source != 255, axis=2)))
    outer = [np.float32(labels == k) for k in (1, 2)]
    inner = [np.float32(np.all(source == c, axis=2)) for c in colors]
    midpoint = (centers[1] + centers[2]) / 2
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(80):
        if frame == 0:
            result = source
        else:
            t = frame / 79
            masks, fills = [], []
            for j in range(2):
                delta = (midpoint - centers[j + 1]) * t
                matrix = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
                masks.append(cv2.warpAffine(outer[j], matrix, (1024, 1024), flags=cv2.INTER_LINEAR))
                fills.append(cv2.warpAffine(inner[j], matrix, (1024, 1024), flags=cv2.INTER_LINEAR))
            a, b = masks
            union = np.maximum(a, b)
            fill = np.maximum(fills[0], fills[1])
            # Add the two light colors where both disks are present.
            total = a[..., None] * colors[0] + b[..., None] * colors[1]
            color = np.minimum(total / np.maximum(union[..., None], 1e-6), 255)
            result = np.uint8(np.clip(255 * (1 - union[..., None]) + color * fill[..., None], 0, 255).round())
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
