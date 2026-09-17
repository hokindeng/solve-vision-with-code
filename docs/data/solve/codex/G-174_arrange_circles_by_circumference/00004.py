from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.any(original != 255, axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    circles = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        circles.append(dict(x=x, y=y, w=w, h=h,
                            sprite=original[y:y+h, x:x+w].copy(),
                            mask=labels[y:y+h, x:x+w] == label))
    circles.sort(key=lambda c: -c['w'])
    gap = 20
    row_width = sum(c['w'] for c in circles) + gap * (len(circles)-1)
    cursor = (1024-row_width)/2
    for c in circles:
        c['tx'] = cursor
        c['ty'] = 512-(c['h']-1)/2
        cursor += c['w']+gap
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output/'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(80):
        t = frame/79
        ease = t*t*t*(10-15*t+6*t*t)
        canvas = np.full_like(original, 255)
        for c in circles:
            x = round(c['x']+(c['tx']-c['x'])*ease)
            y = round(c['y']+(c['ty']-c['y'])*ease)
            region = canvas[y:y+c['h'], x:x+c['w']]
            region[c['mask']] = c['sprite'][c['mask']]
        if frame == 0:
            assert np.array_equal(canvas, original)
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
