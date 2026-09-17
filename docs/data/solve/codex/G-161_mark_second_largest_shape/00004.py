from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    rgb = np.array(base)
    # Rank the connected colored interiors by area.
    mask = ((rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int)) > 40).astype('uint8')
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    shapes = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    idx = shapes[1]
    cx, cy = centers[idx]
    radius = max(stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]) / 2 + 10
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    scale = 4
    for frame in range(40):
        result = base.copy()
        if frame > 0:
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            angle = 360 * frame / 39
            bounds = tuple(v*scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(bounds, -90, -90+angle, fill=(235, 20, 30, 255), width=5*scale)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            result.paste(overlay, (0, 0), overlay)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
