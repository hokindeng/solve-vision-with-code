from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    shape = np.all(base == (25, 25, 255), axis=2).astype(np.uint8)
    contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    # Start at the top vertex and travel clockwise along the actual pixel boundary.
    start = np.lexsort((contour[:, 0], contour[:, 1]))[0]
    contour = np.roll(contour, -start, axis=0)
    contour = np.concatenate((contour[:1], contour[:0:-1], contour[:1]))
    lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    distances = np.r_[0, np.cumsum(lengths)]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format',
           'rgb24', '-video_size', '1024x1024', '-framerate', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '10',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in range(40):
        result = base.copy()
        if frame:
            count = min(len(contour), np.searchsorted(distances, distances[-1] * frame / 39) + 1)
            stroke = np.zeros(shape.shape, np.uint8)
            cv2.polylines(stroke, [contour[:count]], False, 255, 11, cv2.LINE_8)
            # Inset the outline so it cannot change neighboring shapes or background.
            result[(stroke > 0) & (shape > 0)] = (255, 0, 0)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
