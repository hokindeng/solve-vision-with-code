from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.asarray(base)
    mask = np.all(a == (25, 255, 255), axis=2)
    ys, xs = np.where(mask)
    top_y, bottom_y = int(ys.min()), int(ys.max())
    top_x = float(np.where(mask[top_y])[0].mean())
    bottom_x = np.where(mask[bottom_y])[0]
    points = [(top_x, top_y), (int(bottom_x.max()), bottom_y),
              (int(bottom_x.min()), bottom_y), (top_x, top_y)]
    lengths = [np.hypot(q[0]-p[0], q[1]-p[1]) for p,q in zip(points, points[1:])]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for frame in range(40):
        im = base.copy()
        if frame:
            draw = ImageDraw.Draw(im)
            remaining = sum(lengths) * frame / 39
            for p, q, length in zip(points, points[1:], lengths):
                if remaining <= 0:
                    break
                t = min(1.0, remaining / length)
                end = (p[0]+t*(q[0]-p[0]), p[1]+t*(q[1]-p[1]))
                draw.line([p, end], fill=(255,0,0), width=6)
                remaining -= length
        process.stdin.write(im.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
