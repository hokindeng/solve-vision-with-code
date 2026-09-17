from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.array(base)
    # Locate the second largest colored connected component by its area.
    import cv2
    mask = ((arr.max(axis=2) - arr.min(axis=2)) > 30).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    components = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    target = components[1]
    cx, cy = centers[target]
    radius = max(stats[target, cv2.CC_STAT_WIDTH], stats[target, cv2.CC_STAT_HEIGHT]) / 2 + 9
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    scale = 4
    for frame in range(40):
        image = base.copy()
        if frame > 0:
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            progress = min(frame / 37, 1)
            box = tuple(v*scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90+360*progress, fill=(235, 20, 30, 255), width=5*scale)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            image.paste(overlay, (0, 0), overlay)
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
