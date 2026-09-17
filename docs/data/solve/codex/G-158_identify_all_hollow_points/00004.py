from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    rgb = np.array(original)
    dark = (np.min(rgb, axis=2) < 100).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(dark, 8)
    hollow = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w > 50 and h > 50 and area / (w * h) < 0.25:
            cx, cy = centers[i]
            hollow.append((cx, cy, max(w, h) / 2 + 9))
    hollow.sort(key=lambda p: (p[1], p[0]))
    assert len(hollow) == 4, hollow
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
           '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    scale = 3
    for frame in range(80):
        mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
        draw = ImageDraw.Draw(mask)
        for j, (cx, cy, radius) in enumerate(hollow):
            progress = max(0, min(1, (frame - (5 + j * 18)) / 16))
            if progress <= 0:
                continue
            box = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, start=-90, end=-90 + 360 * progress, fill=255, width=4 * scale)
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
        result = Image.composite(Image.new('RGB', original.size, (255, 0, 0)), original, mask)
        proc.stdin.write(np.asarray(result).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    print('Identified hollow circles:', hollow)

if __name__ == '__main__':
    main()
