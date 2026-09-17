from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    centers = [143, 225, 307, 389, 471, 552, 634, 717, 798, 881]
    # Reuse the original raster artwork for both states, including its borders.
    radius = 30
    y = 512
    def patch(x):
        return original[y-radius:y+radius+1, x-radius:x+radius+1].copy()
    on, off = patch(307), patch(143)
    changes = [(0, on), (1, on), (4, off), (5, off), (6, off), (8, off)]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
               'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(output / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_number in range(35):
        frame = original.copy()
        for order, (index, target) in enumerate(changes):
            start = 2 + order * 5
            amount = np.clip((frame_number - start) / 4.0, 0.0, 1.0)
            if amount == 0:
                continue
            x = centers[index]
            before = patch(x)
            blended = np.rint(before.astype(float) * (1-amount) + target.astype(float) * amount).astype(np.uint8)
            frame[y-radius:y+radius+1, x-radius:x+radius+1] = blended
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
