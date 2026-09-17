from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # The purple marker at (773, 800) is the minimum, with value 28.
    corners = [(749, 780), (798, 780), (798, 824), (749, 824), (749, 780)]
    lengths = [49, 44, 49, 44]
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUT / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(48):
        img = base.copy()
        draw = ImageDraw.Draw(img)
        remaining = sum(lengths) * min(frame / 44, 1.0)
        for a, b, length in zip(corners, corners[1:], lengths):
            if remaining <= 0:
                break
            fraction = min(remaining / length, 1.0)
            end = (round(a[0] + (b[0] - a[0]) * fraction),
                   round(a[1] + (b[1] - a[1]) * fraction))
            draw.line([a, end], fill=(255, 0, 0), width=4)
            remaining -= length
        encoder.stdin.write(img.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
