from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(base)
    dark = pixels.min(axis=2) < 100
    peaks = []
    for left, right in [(200, 300), (510, 600), (820, 900)]:
        ys, xs = np.where(dark[:, left:right])
        top = ys.min()
        peaks.append((float(np.mean(xs[ys == top] + left)), float(top + 1)))
    scale = 4
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '10', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in range(10):
        overlay = Image.new('RGBA', (1024 * scale, 1024 * scale))
        draw = ImageDraw.Draw(overlay)
        for index, (x, y) in enumerate(peaks):
            progress = min(1.0, max(0.0, (frame - index * 3) / 3))
            if progress <= 0:
                continue
            radius = 24
            box = tuple(round(v * scale) for v in (x-radius, y-radius, x+radius, y+radius))
            draw.arc(box, -90, -90 + 360 * progress, fill=(235, 0, 0, 255), width=3*scale)
            if progress == 1:
                dot = 4
                draw.ellipse(tuple(round(v * scale) for v in (x-dot,y-dot,x+dot,y+dot)), fill=(235,0,0,255))
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        result = Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    error = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
