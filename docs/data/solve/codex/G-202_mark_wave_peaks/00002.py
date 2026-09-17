from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(original)
    dark = pixels.max(axis=2) < 100
    # Locate the uppermost wave point within each separated crest.
    peaks = []
    for left, right in [(70, 150), (450, 530), (830, 910)]:
        ys, xs = np.where(dark[:, left:right])
        top = int(ys.min())
        peaks.append((float(np.mean(xs[ys == top] + left)), top))
    scale = 4
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(10):
        overlay = Image.new('RGBA', (1024 * scale, 1024 * scale))
        draw = ImageDraw.Draw(overlay)
        for index, (x, y) in enumerate(peaks):
            progress = min(1.0, max(0.0, (frame - index * 3) / 3))
            if progress <= 0:
                continue
            radius = 22
            box = tuple(v * scale for v in (x-radius, y-radius, x+radius, y+radius))
            draw.arc(box, -90, -90 + 360 * progress, fill=(255, 0, 0, 255), width=3*scale)
            dot = 4
            draw.ellipse(tuple(v * scale for v in (x-dot, y-dot, x+dot, y+dot)), fill=(255, 0, 0, 255))
        overlay = overlay.resize(original.size, Image.Resampling.LANCZOS)
        result = Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
