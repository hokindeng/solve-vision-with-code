from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    inside = labels == labels[520, 912]
    ys, _ = np.where(inside)
    top, bottom = ys.min(), ys.max()
    blue = original[520, 112].astype(float)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    rows = np.indices(inside.shape)[0]
    for i in range(64):
        frame = original.copy()
        progress = np.clip((i - 4) / 55, 0, 1)
        # The color rises through only the enclosed white interior.
        boundary = bottom + 1 - progress * (bottom - top + 2)
        alpha = np.clip(rows - boundary + 1, 0, 1)
        a = alpha[inside, None]
        frame[inside] = np.rint(255 * (1-a) + blue * a).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
