from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    width, height = original.size
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # The only five-sided polygon is the olive shape in the upper right.
    cx, cy = 794, 155
    scale = 4
    for frame_index in range(30):
        frame = original.copy()
        if frame_index:
            t = frame_index / 29
            radius = 115 * (t * t * (3 - 2 * t))
            mask = Image.new('L', (width * scale, height * scale), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse(tuple(round(v * scale) for v in
                               (cx-radius, cy-radius, cx+radius, cy+radius)),
                         outline=255, width=5 * scale)
            mask = mask.resize(original.size, Image.Resampling.LANCZOS)
            frame.paste((255, 0, 0), (0, 0), mask)
        encoder.stdin.write(np.asarray(frame).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
