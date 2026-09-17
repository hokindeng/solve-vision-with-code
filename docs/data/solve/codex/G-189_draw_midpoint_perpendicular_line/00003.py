from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # The green point (239, 559) is at the vertical midpoint of the parallels.
    # Draw behind the existing point and meet the inside edges of both lines.
    x, top, bottom = 239, 381, 739
    point = (base.max(axis=2).astype(int) - base.min(axis=2).astype(int)) > 30
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(50):
        frame = base.copy()
        if i:
            end = top + int(round((bottom - top + 1) * i / 49))
            region = frame[top:end, x-2:x+2]
            region[~point[top:end, x-2:x+2]] = (255, 0, 0)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
