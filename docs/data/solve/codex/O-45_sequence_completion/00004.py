from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    src = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The three-shape cycle is heptagon, star, triangle.
    # Copy the existing star exactly, translating it three sequence spacings.
    star = src[450:573, 278:391].copy()
    x, y = 795, 450
    h, w = star.shape[:2]
    original = src[y:y+h, x:x+w].copy()
    out = ROOT / 'output/video.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    for i in range(25):
        frame = src.copy()
        t = i / 24
        if t <= 0.5:
            a = t * 2
            a = a*a*(3-2*a)
            region = original.astype(float)*(1-a) + 255*a
        else:
            a = (t-0.5)*2
            a = a*a*(3-2*a)
            region = 255*(1-a) + star.astype(float)*a
        frame[y:y+h, x:x+w] = np.rint(region).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
