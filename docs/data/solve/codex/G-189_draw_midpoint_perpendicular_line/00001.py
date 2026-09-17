from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    original = np.array(base)
    # The lime point (465, 738) lies halfway between y=621.5 and y=855.5.
    # Preserve the point itself in front of the constructed line.
    point_mask = np.zeros((1024, 1024), dtype=bool)
    point_mask[730:747, 457:474] = (original[730:747, 457:474].max(axis=2) - original[730:747, 457:474].min(axis=2)) > 80
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    for i in range(50):
        frame = base.copy()
        if i:
            end_y = 622 + round((855 - 622) * i / 49)
            ImageDraw.Draw(frame).line([(465, 622), (465, end_y)], fill=(255, 0, 0), width=4)
        arr = np.array(frame)
        arr[point_mask] = original[point_mask]
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
