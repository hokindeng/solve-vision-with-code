from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    pink = np.all(original == (200, 80, 120), axis=2)
    # Trace clockwise from the top. Clip the stroke inside the existing shape
    # so no background pixels or unrelated elements are modified.
    vertices = np.array([[723,206], [808,268], [775,368], [670,368], [637,268], [723,206]], dtype=float)
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    perimeter = lengths.sum()
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(80):
        frame = original.copy()
        if i > 0:
            remaining = perimeter * i / 79
            stroke = np.zeros(pink.shape, dtype=np.uint8)
            for a, b, length in zip(vertices[:-1], vertices[1:], lengths):
                if remaining <= 0:
                    break
                end = a + (b-a) * min(remaining / length, 1)
                cv2.line(stroke, tuple(a.astype(int)), tuple(np.rint(end).astype(int)), 255, 5, cv2.LINE_8)
                remaining -= length
            frame[(stroke > 0) & pink] = 0
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
