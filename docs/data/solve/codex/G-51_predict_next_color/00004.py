from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')
def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    white = np.all(original == 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    interior = labels == labels[510, 912]
    assert not interior[0, 0]
    yy = np.indices(interior.shape)[0]
    ymin, ymax = yy[interior].min(), yy[interior].max()
    teal = np.array([6, 99, 131], dtype=np.uint8)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(64):
        frame = original.copy()
        progress = np.clip((index - 3) / 57, 0, 1)
        threshold = ymax + 1 - progress * (ymax - ymin + 2)
        mask = interior & (yy >= threshold)
        frame[mask] = teal
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
