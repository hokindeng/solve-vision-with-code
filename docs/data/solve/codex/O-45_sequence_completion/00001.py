from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    completed = original.copy()
    # Reuse the existing pink circle exactly, one full four-color cycle later.
    # Circle centers are spaced about 143.75 pixels apart.
    completed[460:564, 819:923] = original[460:564, 244:348]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(25):
        t = index / 24
        weight = t * t * (3 - 2 * t)
        frame = original.copy()
        region = np.s_[460:564, 819:923]
        frame[region] = np.rint(original[region].astype(float) * (1-weight)
                                + completed[region].astype(float) * weight).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
