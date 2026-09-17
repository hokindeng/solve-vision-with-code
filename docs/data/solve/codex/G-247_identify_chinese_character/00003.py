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
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    scale = 4
    for frame in range(48):
        image = base.copy()
        if frame > 0:
            progress = min(frame / 45, 1.0)
            mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
            draw = ImageDraw.Draw(mask)
            # The only Chinese character is 美, centered near (588, 704).
            box = tuple(int(v * scale) for v in (500, 616, 676, 792))
            draw.arc(box, -90, -90 + 360 * progress, fill=255, width=5 * scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            image.paste((230, 25, 30), (0, 0), mask)
        encoder.stdin.write(np.asarray(image).tobytes())
    encoder.stdin.close()
    error = encoder.stderr.read()
    if encoder.wait() != 0:
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
