from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    width, height = source.size
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', '16', '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    scale = 4
    for index in range(16):
        frame = source.copy()
        if index:
            mask = Image.new('L', (width * scale, height * scale), 0)
            draw = ImageDraw.Draw(mask)
            cx, cy, radius = 260, 336, 91
            bounds = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(bounds, start=-90, end=-90 + 360 * index / 15,
                     fill=255, width=4 * scale)
            mask = mask.resize(source.size, Image.Resampling.LANCZOS)
            frame.paste((235, 20, 30), (0, 0), mask)
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
