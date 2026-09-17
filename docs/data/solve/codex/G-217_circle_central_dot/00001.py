from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = (base.min(axis=2) < 80).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    dots = [centroids[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100]
    dots.sort(key=lambda p: p[0])
    cx, cy = dots[len(dots)//2]
    height, width = base.shape[:2]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '10', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    radius = 27
    for frame in range(22):
        img = base.copy()
        if frame:
            overlay = Image.new('L', (width*scale, height*scale))
            draw = ImageDraw.Draw(overlay)
            end = -90 + 360 * frame / 21
            box = tuple(v*scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, end, fill=255, width=4*scale)
            alpha = np.array(overlay.resize((width,height), Image.Resampling.LANCZOS)).astype(float)/255
            a = alpha[...,None]
            img = np.rint(base*(1-a) + np.array([230,25,35])*a).astype(np.uint8)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
