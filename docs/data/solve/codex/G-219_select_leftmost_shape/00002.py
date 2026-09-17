from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    w, h = base.size
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    scale = 4
    cx, cy, radius = 596.5, 185, 80
    for i in range(48):
        frame = base.copy()
        if i > 0:
            mask = Image.new('L', (w * scale, h * scale), 0)
            draw = ImageDraw.Draw(mask)
            progress = i / 47
            angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress,
                                 max(2, int(700 * progress)))
            points = [((cx + radius*np.cos(a))*scale,
                       (cy + radius*np.sin(a))*scale) for a in angles]
            draw.line(points, fill=255, width=5*scale)
            for x, y in (points[0], points[-1]):
                draw.ellipse((x-2.5*scale, y-2.5*scale,
                              x+2.5*scale, y+2.5*scale), fill=255)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            frame.paste((235, 25, 35), (0, 0), mask)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
