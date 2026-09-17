from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path('/app')

def main():
    src = cv2.imread(str(ROOT / 'first_frame.png'))
    h, w = src.shape[:2]
    foreground = np.any(src != 255, axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    ids = sorted(range(1, count), key=lambda i: (int(centers[i][1] // 200), centers[i][0]))
    yy, xx = np.indices((h, w))
    borders = []
    for i in ids:
        mask = (labels == i).astype(np.uint8)
        # The three-pixel stroke lies inside the shape, preserving background pixels.
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        edge = (mask > 0) & (inner == 0)
        cx, cy = centers[i]
        phase = np.mod(np.arctan2(xx-cx, -(yy-cy)), 2*np.pi)/(2*np.pi)
        borders.append((edge, phase))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(80):
        image = src.copy()
        progress = frame / 79 * len(borders)
        for index, (edge, phase) in enumerate(borders):
            portion = np.clip(progress-index, 0, 1)
            if portion > 0:
                image[edge & (phase < portion)] = 0
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
