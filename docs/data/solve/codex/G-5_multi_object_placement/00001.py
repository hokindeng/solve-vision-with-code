from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.uint8(np.any(original != 255, axis=2))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    objects, markers = [], []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        color = tuple(original[labels == label][0])
        item = (label, x, y, w, h, centers[label], color)
        (objects if area == w * h else markers).append(item)
    background = original.copy()
    for label, *_ in objects:
        background[labels == label] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '12', '-preset', 'slow',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(48):
        t = frame_index / 47
        frame = background.copy()
        for label, x, y, w, h, start, color in objects:
            target = next(m[5] for m in markers if m[6] == color)
            center = start + t * (target - start)
            left, top = np.rint(center - np.array([(w-1)/2, (h-1)/2])).astype(int)
            frame[top:top+h, left:left+w] = color
        if frame_index == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
