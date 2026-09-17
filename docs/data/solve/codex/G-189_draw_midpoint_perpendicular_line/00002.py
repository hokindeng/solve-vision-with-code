from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(base)
    # Locate the two horizontal rules and the marker nearest their midpoint.
    rows = np.flatnonzero(np.sum(np.all(pixels < 30, axis=2), axis=1) > 900)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    top, bottom = (float(np.mean(g)) for g in groups)
    middle = (top + bottom) / 2
    marker = (pixels[:,:,0] > 100) & (pixels[:,:,0] < 230) & (np.max(pixels, axis=2) - np.min(pixels, axis=2) < 3)
    yy, xx = np.where(marker & (np.abs(np.arange(base.height)[:,None] - middle) < 15))
    x = float(np.mean(xx))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(50):
        image = base.copy()
        if frame:
            progress = frame / 49
            end = top + (bottom - top) * progress
            ImageDraw.Draw(image).line([(round(x), round(top)), (round(x), round(end))], fill=(255,0,0), width=4)
        encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
