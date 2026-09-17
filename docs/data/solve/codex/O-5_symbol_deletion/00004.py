from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    # This cell contains only the marked triangle and its deletion border.
    # Keep the remaining cells in their exact original positions.
    ys, xs = slice(446, 579), slice(446, 579)
    cell = original[ys, xs].astype(np.float64)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(46):
        frame = original.copy()
        # Brief recognition pause, then a smooth disappearance across the clip.
        progress = np.clip((index - 6) / 33.0, 0.0, 1.0)
        alpha = progress * progress * (3.0 - 2.0 * progress)
        frame[ys, xs] = np.rint(cell * (1.0 - alpha) + 255.0 * alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    error = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
