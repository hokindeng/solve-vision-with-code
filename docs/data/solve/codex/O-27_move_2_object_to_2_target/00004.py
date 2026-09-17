from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original[0, 0].copy()
    foreground = np.any(original != background, axis=2).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    objects = []
    base = original.copy()
    for point, destination_delta in [((560, 81), (-104, 350)), ((613, 918), (-324, -26))]:
        x, y = point
        label = labels[y, x]
        mask = labels == label
        ys, xs = np.where(mask)
        colors = original[ys, xs].copy()
        objects.append((ys, xs, colors, destination_delta))
        base[mask] = background
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(35):
        frame = base.copy()
        t = index / 34
        for ys, xs, colors, (dx, dy) in objects:
            frame[ys + round(dy*t), xs + round(dx*t)] = colors
        if index == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    error = encoder.stderr.read()
    if encoder.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
