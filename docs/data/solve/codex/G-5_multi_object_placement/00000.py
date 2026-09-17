from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    objects = []
    stars = np.zeros(original.shape[:2], dtype=bool)
    for color in [(255, 100, 100), (255, 100, 255)]:
        mask = np.all(original == color, axis=2).astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        ids = sorted(range(1, len(stats)), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        star, obj = ids
        stars |= labels == star
        x, y, w, h, _ = stats[obj]
        sprite = (labels[y:y+h, x:x+w] == obj).astype(np.uint8) * 255
        center = np.array([x + (w-1)/2, y + (h-1)/2])
        delta = centroids[star] - center
        objects.append((color, sprite, x, y, delta))
        background[labels == obj] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(48):
        frame = background.copy()
        t = i / 47
        for color, sprite, x, y, delta in objects:
            dx, dy = delta * t
            transform = np.float32([[1, 0, x + dx], [0, 1, y + dy]])
            alpha = cv2.warpAffine(sprite, transform, (1024, 1024), flags=cv2.INTER_LINEAR)
            active = alpha > 0
            a = alpha[active, None].astype(np.float32) / 255
            frame[active] = np.rint(frame[active] * (1-a) + np.array(color)*a).astype(np.uint8)
        frame[stars] = original[stars]
        if i == 0:
            frame = original
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
