from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # The top-right room contains the toilet, basin, and bathtub.
    corners = [(699, 61), (964, 61), (964, 309), (699, 309), (699, 61)]
    lengths = [abs(b[0]-a[0])+abs(b[1]-a[1]) for a,b in zip(corners,corners[1:])]
    perimeter = sum(lengths)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(28):
        frame = source.copy()
        if index:
            draw = ImageDraw.Draw(frame)
            remaining = perimeter * index / 27
            for a, b, length in zip(corners, corners[1:], lengths):
                if remaining <= 0:
                    break
                fraction = min(remaining / length, 1)
                end = (round(a[0]+(b[0]-a[0])*fraction), round(a[1]+(b[1]-a[1])*fraction))
                draw.line([a, end], fill=(0, 190, 55), width=7)
                remaining -= length
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
