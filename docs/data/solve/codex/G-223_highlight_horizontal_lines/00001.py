from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    base = np.array(source)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # The sole horizontal segment runs from (570, 270) to (733, 270).
    # A circular outline encloses it without touching any existing line.
    cx, cy, radius = 651.5, 269.5, 98
    scale = 4
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(48):
        if frame == 0:
            result = base
        else:
            progress = min(frame / 45.0, 1.0)
            mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
            draw = ImageDraw.Draw(mask)
            box = tuple(int(v * scale) for v in
                        (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, start=-90, end=-90 + 360*progress,
                     fill=255, width=4*scale)
            mask = mask.resize(source.size, Image.Resampling.LANCZOS)
            alpha = np.asarray(mask, dtype=np.float32)[:, :, None] / 255
            result = np.rint(base * (1-alpha)).astype(np.uint8)
        encoder.stdin.write(result.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
