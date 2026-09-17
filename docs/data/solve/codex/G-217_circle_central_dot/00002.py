from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.array(base)
    count, labels, stats, centers = cv2.connectedComponentsWithStats((a.max(axis=2) < 80).astype(np.uint8))
    dots = sorted([centers[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100], key=lambda p: p[0])
    cx, cy = dots[len(dots)//2]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1024x1024', '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    radius = 24
    for i in range(22):
        frame = base.copy()
        if i:
            mask = Image.new('L', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(mask)
            angle = 360 * i/21
            box = tuple(int(v*scale) for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90+angle, fill=255, width=3*scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste((255,0,0), (0,0), mask)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
