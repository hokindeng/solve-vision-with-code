from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    points = [(445, 862), (193, 534), (523, 504), (534, 164)]
    yy, xx = np.indices(base.shape[:2])
    protected = np.zeros(base.shape[:2], dtype=bool)
    for x, y in points:
        protected |= (xx-x)**2 + (yy-y)**2 <= 46**2
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(55):
        canvas = base.copy()
        for segment in range(3):
            progress = np.clip((frame - segment * 18) / 18, 0, 1)
            if progress <= 0:
                continue
            start = np.array(points[segment], dtype=float)
            end = start + progress * (np.array(points[segment+1]) - start)
            cv2.line(canvas, tuple(start.astype(int)), tuple(np.rint(end).astype(int)),
                     (255, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        canvas[protected] = base[protected]
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
