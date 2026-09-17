from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Reuse the source's exact raster shapes and colors, including its glow.
    centers = [170, 307, 443, 580, 716, 853]
    top, bottom, radius = 466, 558, 46
    on = original[top:bottom, centers[5]-radius:centers[5]+radius+1].copy()
    off = original[top:bottom, centers[0]-radius:centers[0]+radius+1].copy()
    changes = [(1, 3, 8, on), (2, 10, 15, on), (3, 17, 22, on), (5, 25, 30, off)]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for t in range(35):
        frame = original.copy()
        for index, start, end, target in changes:
            amount = np.clip((t-start)/(end-start), 0.0, 1.0)
            amount = amount*amount*(3-2*amount)
            x = centers[index]
            source = original[top:bottom, x-radius:x+radius+1]
            frame[top:bottom, x-radius:x+radius+1] = np.rint(
                source.astype(float)*(1-amount) + target.astype(float)*amount
            ).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
