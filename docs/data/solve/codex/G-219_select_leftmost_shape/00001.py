from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.array(base)
    mask = (np.min(arr, axis=2) < 245).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100]
    target = min(candidates, key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    x, y, w, h, _ = stats[target]
    cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
    radius = max(w, h) / 2 + 10
    scale = 4
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '12',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(48):
        frame = base.copy()
        if frame_index:
            overlay = Image.new('RGBA', (1024 * scale, 1024 * scale))
            draw = ImageDraw.Draw(overlay)
            progress = frame_index / 47
            angles = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * progress,
                                 max(2, int(700 * progress)))
            points = [((cx + radius * np.cos(a)) * scale,
                       (cy + radius * np.sin(a)) * scale) for a in angles]
            draw.line(points, fill=(255, 0, 0, 255), width=4 * scale)
            for px, py in (points[0], points[-1]):
                draw.ellipse((px-2*scale, py-2*scale, px+2*scale, py+2*scale),
                             fill=(255, 0, 0, 255))
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0, 0), overlay)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
