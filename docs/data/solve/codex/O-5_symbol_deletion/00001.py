from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    # Only the marked tile changes. All other pixels retain their source values.
    x0, y0, x1, y1 = 378, 446, 511, 579
    tile = original[y0:y1, x0:x1].astype(np.float64)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '12',
           '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(output / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(46):
        frame = original.copy()
        # Briefly establish the marked square, then erase it smoothly.
        t = np.clip((i - 5) / 35.0, 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)
        frame[y0:y1, x0:x1] = np.rint(tile * (1-alpha) + 255 * alpha).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
