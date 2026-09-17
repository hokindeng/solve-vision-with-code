from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    gray = np.asarray(original.convert('L'))
    count, labels, stats, centers = cv2.connectedComponentsWithStats((gray < 128).astype(np.uint8))
    hollow = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w > 20 and h > 20 and .8 < w / h < 1.2 and area / (w * h) < .25:
            hollow.append((float(centers[i, 0]), float(centers[i, 1]), max(w, h) / 2 + 10))
    hollow.sort(key=lambda p: (p[1], p[0]))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    for frame in range(80):
        # Pause to inspect the points, then trace each identified hollow point.
        progress = np.clip((frame - 10) / 59, 0, 1) * len(hollow)
        overlay = Image.new('RGBA', (1024 * scale, 1024 * scale))
        draw = ImageDraw.Draw(overlay)
        for j, (cx, cy, radius) in enumerate(hollow):
            fraction = float(np.clip(progress - j, 0, 1))
            if fraction <= 0:
                continue
            box = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90 + 360 * fraction, fill=(235, 25, 35, 255), width=4 * scale)
        overlay = overlay.resize(original.size, Image.Resampling.LANCZOS)
        result = Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')
        proc.stdin.write(np.asarray(result).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
