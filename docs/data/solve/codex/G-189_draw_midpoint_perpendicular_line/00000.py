from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    dark = (base.max(axis=2) < 40)
    rows = np.flatnonzero(dark.sum(axis=1) > 900)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    upper, lower = [float(g.mean()) for g in groups]
    midpoint = (upper + lower) / 2
    colored = ((base.max(axis=2).astype(int) - base.min(axis=2).astype(int)) > 40).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(colored)
    points = [centers[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 20]
    x, _ = min(points, key=lambda p: abs(p[1] - midpoint))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(50):
        im = Image.fromarray(base.copy())
        if frame:
            progress = frame / 49
            end = upper + (lower - upper) * progress
            ImageDraw.Draw(im).line((round(x), round(upper), round(x), round(end)), fill=(255, 0, 0), width=4)
        proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
