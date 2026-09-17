from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    completed = original.copy()
    # Copy the second circle exactly to the sixth position (center x=871).
    completed[463:561, 822:921] = original[463:561, 247:346]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(25):
        t = index / 24
        alpha = t * t * (3 - 2 * t)
        frame = original.copy()
        frame[463:561, 822:921] = np.rint(
            original[463:561, 822:921].astype(float) * (1 - alpha)
            + completed[463:561, 822:921].astype(float) * alpha
        ).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
