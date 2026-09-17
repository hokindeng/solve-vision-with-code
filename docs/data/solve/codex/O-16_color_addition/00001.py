from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    ids = sorted(range(1, len(stats)), key=lambda i: centers[i, 0])
    midpoint = np.mean(centers[ids], axis=0)
    objects = []
    for i in ids:
        silhouette = (labels == i).astype(np.uint8)
        inside = (silhouette.astype(bool) & np.any(original != 0, axis=2)).astype(np.uint8)
        color = original[inside.astype(bool)][0].astype(np.uint16)
        objects.append((silhouette, inside, color, midpoint - centers[i]))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
        '-i', '-', '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(80):
        if frame == 0:
            result = original
        else:
            progress = frame / 79
            masks, interiors = [], []
            for silhouette, inside, color, displacement in objects:
                dx, dy = displacement * progress
                transform = np.float32([[1, 0, dx], [0, 1, dy]])
                masks.append(cv2.warpAffine(silhouette, transform, (1024, 1024), flags=cv2.INTER_NEAREST).astype(bool))
                interiors.append(cv2.warpAffine(inside, transform, (1024, 1024), flags=cv2.INTER_NEAREST).astype(bool))
            result = np.full_like(original, 255)
            result[masks[0] | masks[1]] = 0
            result[interiors[0] & ~masks[1]] = objects[0][2]
            result[interiors[1] & ~masks[0]] = objects[1][2]
            overlap = masks[0] & masks[1] & (interiors[0] | interiors[1])
            result[overlap] = np.minimum(objects[0][2] + objects[1][2], 255).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
