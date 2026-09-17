from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    objects, markers = [], []
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        mask = labels == k
        color = tuple(original[mask][0])
        item = (mask, np.array([x + (w-1)/2, y + (h-1)/2]), color)
        (objects if area > 1000 else markers).append(item)
    background = original.copy()
    movements = []
    for mask, center, color in objects:
        background[mask] = 255
        target = next(c for _, c, col in markers if col == color)
        ys, xs = np.nonzero(mask)
        movements.append((ys, xs, target-center, color))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(48):
        frame = background.copy()
        progress = frame_index / 47
        for ys, xs, delta, color in movements:
            dx, dy = np.rint(delta * progress).astype(int)
            frame[ys + dy, xs + dx] = color
        if frame_index == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
