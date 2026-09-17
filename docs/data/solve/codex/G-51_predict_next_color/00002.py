from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    white = np.all(base == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white)
    inside = labels == labels[512, 912]
    ys = np.where(inside)[0]
    top, bottom = ys.min(), ys.max()
    blue = base[512, 112].astype(float)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    rows = np.arange(1024)[:, None]
    for i in range(64):
        frame = base.copy()
        if i:
            t = min(i / 60.0, 1.0)
            progress = t * t * (3 - 2 * t)
            level = bottom + 1 - progress * (bottom - top + 2)
            amount = np.clip(rows - level + 1, 0, 1)
            alpha = np.broadcast_to(amount, inside.shape)[inside, None]
            frame[inside] = np.rint(base[inside] * (1 - alpha) + blue * alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
