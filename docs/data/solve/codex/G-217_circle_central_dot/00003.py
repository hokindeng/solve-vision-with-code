from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import cv2

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.array(original)
    mask = (pixels.max(axis=2) < 100).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    dots = sorted([centers[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100], key=lambda c: c[0])
    cx, cy = dots[len(dots)//2]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    for index in range(22):
        frame = original.copy()
        if index:
            # Animate only the new ring; the source image is never resampled.
            overlay_mask = Image.new('L', (1024*scale, 1024*scale), 0)
            draw = ImageDraw.Draw(overlay_mask)
            radius = 26.5
            bbox = tuple(round(v*scale) for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(bbox, start=-90, end=-90+360*index/21, fill=255, width=3*scale)
            overlay_mask = overlay_mask.resize(original.size, Image.Resampling.LANCZOS)
            frame.paste((230, 0, 0), (0, 0), overlay_mask)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
