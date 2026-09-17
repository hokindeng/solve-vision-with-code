from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    heart = np.zeros(source.shape[:2], dtype=bool)
    heart[465:560, 413:508] = np.all(source[465:560, 413:508] == (238, 130, 238), axis=2)
    # Copy the exact raster silhouette from the reference into the third box.
    ref = np.all(source[43:112, 912:981] == (0, 0, 255), axis=2)
    diamond = np.zeros_like(heart)
    diamond[478:547, 426:495] = ref
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
               'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(52):
        frame = source.copy()
        if i > 0:
            fade_out = smooth(i / 25)
            frame[heart] = np.rint(np.array([238, 130, 238]) * (1 - fade_out) + 255 * fade_out).astype(np.uint8)
            if i >= 26:
                fade_in = smooth((i - 26) / 25)
                frame[diamond] = np.rint(255 * (1 - fade_in) + np.array([0, 0, 255]) * fade_in).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
