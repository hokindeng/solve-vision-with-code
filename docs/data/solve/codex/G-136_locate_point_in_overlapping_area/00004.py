from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # A point qualifies only if its entire outer neighborhood is overlap green.
    overlap = np.all(base == (97, 200, 69), axis=2)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        (base.max(axis=2) < 30).astype(np.uint8))
    targets = []
    for label in range(1, count):
        point = (labels == label).astype(np.uint8)
        expanded = cv2.dilate(point, np.ones((5, 5), np.uint8)) != 0
        surround = expanded & (point == 0)
        if overlap[surround].all():
            targets.append(tuple(map(float, centers[label])))
    targets.sort(key=lambda p: (p[1], p[0]))
    assert len(targets) == 2, targets
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '10',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(37):
        # Leave an initial viewing interval, then draw each circle in turn.
        mask = Image.new('L', (4096, 4096), 0)
        draw = ImageDraw.Draw(mask)
        for index, (x, y) in enumerate(targets):
            start = 5 + index * 14
            progress = min(1.0, max(0.0, (frame - start) / 12))
            if progress:
                r = 17
                box = tuple(int(v * 4) for v in (x-r, y-r, x+r, y+r))
                draw.arc(box, -90, -90 + 360 * progress, fill=255, width=12)
        mask = np.array(mask.resize((1024, 1024), Image.Resampling.LANCZOS))
        alpha = mask[..., None].astype(np.float32) / 255
        result = np.rint(base * (1-alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    print('Circled points:', targets)

if __name__ == '__main__':
    main()
