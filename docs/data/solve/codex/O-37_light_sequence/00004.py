from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    centers = [143, 225, 307, 389, 471, 552, 634, 717, 799, 880]
    y, r = 512, 30
    # Reuse the exact original raster artwork for both states.
    on = original[y-r:y+r+1, 143-r:143+r+1].copy()
    off = original[y-r:y+r+1, 307-r:307+r+1].copy()
    changes = [(0, off), (1, off), (2, on), (4, off), (5, off), (6, off), (9, off)]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(35):
        frame = original.copy()
        for order, (index, target) in enumerate(changes):
            start = 2 + order * 4
            progress = np.clip((frame_index - start) / 3.0, 0, 1)
            if progress == 0:
                continue
            x = centers[index]
            source = original[y-r:y+r+1, x-r:x+r+1]
            frame[y-r:y+r+1, x-r:x+r+1] = np.rint(
                source.astype(float) * (1-progress) + target.astype(float) * progress
            ).astype(np.uint8)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
