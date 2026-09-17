from pathlib import Path
import subprocess
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'
COLOR = (184, 134, 11)
CIRCLES = ((462, 307, 448), (787, 716, 223))

def frame(t):
    im = Image.new('RGB', (1024, 1024), 'white')
    draw = ImageDraw.Draw(im)
    # Smooth acceleration and deceleration across the entire clip.
    u = t * t * (3 - 2 * t)
    for sx, sy, radius in CIRCLES:
        x = round(sx + (512 - sx) * u)
        y = round(sy + (512 - sy) * u)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     outline=COLOR, width=8)
    return im

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    assert np.array_equal(np.asarray(source), np.asarray(frame(0)))
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT)
    ], stdin=subprocess.PIPE)
    for i in range(40):
        process.stdin.write(frame(i / 39).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
