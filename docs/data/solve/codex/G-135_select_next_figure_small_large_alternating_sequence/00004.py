from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    w, h = base.size
    scale = 4
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(60):
        frame = base.copy()
        if i > 9:
            progress = min(1.0, (i - 9) / 40.0)
            mask = Image.new('L', (w * scale, h * scale), 0)
            draw = ImageDraw.Draw(mask)
            # Circle only the matching large red circle in the fourth card.
            cx, cy, radius = 856, 855, 80
            box = tuple(int(v * scale) for v in
                        (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90 + 360 * progress, fill=255, width=6*scale)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            frame.paste((235, 25, 35), (0, 0), mask)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    error = proc.stderr.read().decode()
    if proc.wait():
        raise RuntimeError(error)

if __name__ == '__main__':
    main()
