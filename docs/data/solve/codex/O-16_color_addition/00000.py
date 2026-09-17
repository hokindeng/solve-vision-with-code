from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    occupied = (original.min(axis=2) < 255).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(occupied)
    objects = sorted(range(1, count), key=lambda k: centers[k, 1])
    masks, fills, colors = [], [], []
    for k in objects:
        mask = labels == k
        pixels, frequencies = np.unique(original[mask], axis=0, return_counts=True)
        color = pixels[frequencies.argmax()]
        masks.append(mask.astype(np.float32))
        fills.append((mask & np.all(original == color, axis=2)).astype(np.float32))
        colors.append(color.astype(np.float32))
    midpoint = centers[objects].mean(axis=0)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '15',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(80):
        if frame_index == 0:
            frame = original
        else:
            # Both centers travel the same distance at a constant speed.
            t = frame_index / 79
            outer, inner = [], []
            for j, k in enumerate(objects):
                dx, dy = (midpoint - centers[k]) * t
                matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                outer.append(cv2.warpAffine(masks[j], matrix, (1024, 1024), flags=cv2.INTER_LINEAR))
                inner.append(cv2.warpAffine(fills[j], matrix, (1024, 1024), flags=cv2.INTER_LINEAR))
            coverage = np.maximum(outer[0], outer[1])
            inside = np.maximum(inner[0], inner[1])
            # Sum the light contributions; retain a black outline on the union.
            light = outer[0][..., None] * colors[0] + outer[1][..., None] * colors[1]
            light = np.minimum(light, 255)
            edge = np.divide(inside, coverage, out=np.zeros_like(inside), where=coverage > 0)
            frame = np.rint(255 * (1 - coverage[..., None]) + light * edge[..., None]).clip(0,255).astype(np.uint8)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
