from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    width, height = source.size
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # The only horizontal segment spans x=295..477 at y=709..714.
    # A circular outline surrounds it without touching the segment.
    scale = 4
    cx, cy, radius = 386.5, 712.0, 105.0
    for frame in range(48):
        image = source.copy()
        if frame:
            progress = min(frame / 46.0, 1.0)
            mask = Image.new('L', (width * scale, height * scale), 0)
            draw = ImageDraw.Draw(mask)
            angles = np.linspace(-np.pi / 2, -np.pi / 2 + 2*np.pi*progress,
                                 max(2, int(800*progress)))
            points = [((cx+radius*np.cos(a))*scale, (cy+radius*np.sin(a))*scale)
                      for a in angles]
            draw.line(points, fill=255, width=4*scale, joint='curve')
            for x, y in (points[0], points[-1]):
                draw.ellipse((x-2*scale, y-2*scale, x+2*scale, y+2*scale), fill=255)
            mask = mask.resize(source.size, Image.Resampling.LANCZOS)
            image.paste((0, 0, 0), (0, 0), mask)
        encoder.stdin.write(np.asarray(image).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
