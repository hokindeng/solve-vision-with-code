from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(original == (255, 255, 0), axis=2)
    yy, xx = np.where(mask)
    background = original.copy()
    background[mask] = (0, 255, 0)
    # Four cells right and eight down: Manhattan distance is 12.
    waypoints = [(102 * c, 0) for c in range(5)]
    waypoints += [(408, 102 * r) for r in range(1, 9)]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(80):
        progress = np.clip((frame - 3) / 72, 0, 1) * 12
        segment = min(int(progress), 11)
        fraction = progress - segment
        a, b = waypoints[segment], waypoints[segment + 1]
        dx, dy = [round(a[k] + (b[k] - a[k]) * fraction) for k in range(2)]
        image = background.copy()
        image[yy + dy, xx + dx] = (255, 255, 0)
        if frame == 0:
            assert np.array_equal(image, original)
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
