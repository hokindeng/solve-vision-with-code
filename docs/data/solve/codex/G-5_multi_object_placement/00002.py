from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    objects = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 1000]
    markers = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] <= 1000]
    background = original.copy()
    sprites = []
    for obj in objects:
        ys, xs = np.where(labels == obj)
        color = original[ys[0], xs[0]]
        marker = next(m for m in markers if np.array_equal(original[labels == m][0], color))
        x, y, w, h, _ = stats[obj]
        center = np.array([x + (w-1)/2, y + (h-1)/2])
        displacement = centers[marker] - center
        sprites.append((ys, xs, color, displacement))
        background[ys, xs] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(48):
        frame = background.copy()
        for ys, xs, color, displacement in sprites:
            dx, dy = np.rint(displacement * index / 47).astype(int)
            frame[ys + dy, xs + dx] = color
        if index == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
