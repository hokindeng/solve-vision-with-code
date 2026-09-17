from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    h, w = base.shape[:2]
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(OUT / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(30):
        frame = base.copy()
        if i:
            t = i / 29
            eased = t * t * (3 - 2 * t)
            radius = 108 * eased
            scale = 4
            mask = Image.new('L', (w * scale, h * scale), 0)
            draw = ImageDraw.Draw(mask)
            cx, cy = 266, 660
            draw.ellipse(tuple(round(v * scale) for v in
                               (cx-radius, cy-radius, cx+radius, cy+radius)),
                         outline=255, width=5 * scale)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255
            frame = np.rint(base * (1-alpha) + np.array([235, 25, 35]) * alpha).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
