from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    src = cv2.imread(str(ROOT / 'first_frame.png'))
    # Extract complete original silhouettes, including their edge pixels.
    foreground = np.any(src != 255, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    objects = []
    base = src.copy()
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 1000 or x > 500:
            continue
        mask = labels[y:y+h, x:x+w] == i
        patch = src[y:y+h, x:x+w].copy()
        displacement = 504 if x < 100 else (274 if y < 400 else 465)
        objects.append((x, y, w, h, mask, patch, displacement))
        base[labels == i] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(30):
        t = frame_index / 29
        progress = t*t*(3-2*t)
        frame = base.copy()
        for x, y, w, h, mask, patch, displacement in objects:
            xx = x + round(displacement * progress)
            region = frame[y:y+h, xx:xx+w]
            region[mask] = patch[mask]
        if frame_index == 0:
            assert np.array_equal(frame, src)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
