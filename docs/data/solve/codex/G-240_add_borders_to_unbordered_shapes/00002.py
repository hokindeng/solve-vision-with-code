from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def draw_border(frame, vertices, progress):
    if progress <= 0:
        return
    points = np.array(vertices + [vertices[0]], dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    remaining = float(lengths.sum()) * min(progress, 1.0)
    for start, end, length in zip(points[:-1], points[1:], lengths):
        if remaining <= 0:
            break
        stop = start + (end - start) * min(remaining / length, 1)
        cv2.line(frame, tuple(np.rint(start).astype(int)),
                 tuple(np.rint(stop).astype(int)), (0, 0, 0), 4, cv2.LINE_AA)
        remaining -= length

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    shapes = [
        [(718, 167), (851, 262), (800, 419), (635, 419), (585, 262)],
        [(306, 603), (416, 683), (374, 811), (237, 811), (195, 683)],
    ]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(80):
        frame = original.copy()
        draw_border(frame, shapes[0], i / 39)
        draw_border(frame, shapes[1], (i - 39) / 40)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
