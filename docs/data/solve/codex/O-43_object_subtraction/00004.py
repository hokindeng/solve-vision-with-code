from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Both green cone silhouettes and the orange cone are the requested objects.
    selected = (np.all(source == (0, 200, 0), axis=2) |
                np.all(source == (255, 128, 0), axis=2))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
           '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(out / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    colors = source[selected].astype(np.float64)
    for index in range(96):
        t = index / 95
        fade = t * t * (3 - 2 * t)
        frame = source.copy()
        frame[selected] = np.rint(colors + (255 - colors) * fade).astype(np.uint8)
        assert np.array_equal(frame[~selected], source[~selected])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
