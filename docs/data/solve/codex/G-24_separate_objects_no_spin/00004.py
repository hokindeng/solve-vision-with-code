from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = cv2.imread(str(ROOT / 'first_frame.png'))
    # Extract the complete original raster silhouettes, including their edges.
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    foreground[:, 500:] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    objects = []
    background = original.copy()
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 100:
            continue
        mask = labels[y:y+h, x:x+w] == label
        pixels = original[y:y+h, x:x+w].copy()
        cx, cy = centers[label]
        if cy < 500:
            distance = 510 if cx < 300 else 440
        else:
            distance = 600 if cx < 300 else 275
        objects.append((x, y, w, h, mask, pixels, distance))
        background[y:y+h, x:x+w][mask] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(30):
        t = index / 29
        progress = t * t * (3 - 2 * t)
        frame = background.copy()
        for x, y, w, h, mask, pixels, distance in objects:
            new_x = x + round(distance * progress)
            frame[y:y+h, new_x:new_x+w][mask] = pixels[mask]
        if index == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
