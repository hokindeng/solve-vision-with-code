from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    scale = 4
    for i in range(60):
        frame = base.copy()
        if i > 11:
            progress = min(1.0, (i - 11) / 40)
            mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
            draw = ImageDraw.Draw(mask)
            # Circle only the large blue square in the first answer card.
            cx, cy, radius = 166.5, 854.5, 76
            box = tuple(round(v * scale) for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, start=-90, end=-90 + 360 * progress, fill=255, width=5*scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste((225, 30, 38), (0, 0), mask)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    error = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
