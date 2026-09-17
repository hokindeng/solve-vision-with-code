from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    width, height = original.size
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    # The sole pentagon is the teal polygon in the upper-left corner.
    cx, cy = 179, 140
    scale = 4
    for index in range(30):
        frame = original.copy()
        if index:
            t = index / 29
            progress = t * t * (3 - 2 * t)
            radius = 83 * progress
            stroke = min(5, radius * 2)
            mask = Image.new('L', (width * scale, height * scale), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse(tuple(round(v * scale) for v in
                               (cx-radius, cy-radius, cx+radius, cy+radius)),
                         outline=255, width=max(1, round(stroke * scale)))
            mask = mask.resize((width, height), Image.Resampling.LANCZOS)
            frame.paste((255, 0, 0), (0, 0), mask)
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
