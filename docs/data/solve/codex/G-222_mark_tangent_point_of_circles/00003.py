from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    (ROOT / 'output').mkdir(exist_ok=True)
    writer = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(ROOT / 'output/video.mp4')
    ], stdin=subprocess.PIPE)
    # The blue and orange disks meet at (290.5, 392.5).
    scale = 4
    cx, cy, radius = 290.5, 392.5, 19
    box = tuple(int(v * scale) for v in
                (cx-radius, cy-radius, cx+radius, cy+radius))
    for frame in range(60):
        image = base.copy()
        if frame > 0:
            progress = min(frame / 56, 1)
            mask = Image.new('L', (1024*scale, 1024*scale), 0)
            draw = ImageDraw.Draw(mask)
            draw.arc(box, start=-90, end=-90 + 360*progress,
                     fill=255, width=4*scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            image.paste((0, 0, 0), (0, 0), mask)
        writer.stdin.write(np.asarray(image).tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
