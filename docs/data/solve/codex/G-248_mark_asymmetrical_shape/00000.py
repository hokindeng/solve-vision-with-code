from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    scale = 4
    for frame in range(16):
        result = base.copy()
        if frame:
            overlay = Image.new('RGBA', (1024 * scale, 1024 * scale))
            draw = ImageDraw.Draw(overlay)
            # The blue polygon alone lacks reflection symmetry.
            bounds = tuple(v * scale for v in (32, 641, 192, 801))
            draw.arc(bounds, start=-90, end=-90 + 360 * frame / 15,
                     fill=(255, 0, 0, 255), width=5 * scale)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            result.paste(overlay, (0, 0), overlay)
        proc.stdin.write(np.asarray(result).tobytes())
    proc.stdin.close()
    errors = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(errors.decode())

if __name__ == '__main__':
    main()
