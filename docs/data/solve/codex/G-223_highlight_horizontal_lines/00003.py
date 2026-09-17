from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    scale = 4
    for i in range(48):
        frame = base.copy()
        if i > 0:
            # Trace one true circle clockwise, with a brief completed hold.
            progress = min(i / 44, 1.0)
            mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
            draw = ImageDraw.Draw(mask)
            cx, cy, radius = 634, 271, 96
            box = tuple(int(v * scale) for v in
                        (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90 + 360 * progress, fill=255, width=4*scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste((0, 0, 0), (0, 0), mask)
        encoder.stdin.write(np.asarray(frame).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
