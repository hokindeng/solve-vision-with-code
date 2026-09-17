from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.asarray(base)
    mask = ((arr[:, :, 2] > 150) & (arr[:, :, 0] < 100)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    # The three squares fill their bounding boxes; the circle has lower fill.
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100]
    unique = min(candidates, key=lambda i: stats[i, 4] / (stats[i, 2] * stats[i, 3]))
    cx, cy = centers[unique]
    radius = max(stats[unique, 2:4]) / 2 + 13
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
           '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in range(60):
        image = base.copy()
        # Pause to inspect the set, draw clockwise, then hold the completed answer.
        if frame >= 10:
            progress = min(1.0, (frame - 9) / 42)
            scale = 4
            overlay = Image.new('RGBA', (1024 * scale, 1024 * scale))
            draw = ImageDraw.Draw(overlay)
            box = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90 + 360 * progress, fill=(235, 25, 35, 255), width=5*scale)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            image.paste(overlay, (0, 0), overlay)
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
