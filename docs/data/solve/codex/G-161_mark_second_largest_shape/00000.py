from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Locate the three filled circles and select the second largest by area.
    colored = ((base.max(axis=2).astype(int) - base.min(axis=2).astype(int)) > 50).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(colored, 8)
    objects = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    selected = objects[1]
    cx, cy = centers[selected]
    radius = max(stats[selected, cv2.CC_STAT_WIDTH], stats[selected, cv2.CC_STAT_HEIGHT]) / 2 + 13
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(40):
        canvas = base.copy()
        if frame:
            scale = 4
            mask = Image.new('L', (1024 * scale, 1024 * scale))
            draw = ImageDraw.Draw(mask)
            bounds = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(bounds, start=-90, end=-90 + 360 * frame / 39, fill=255, width=5*scale)
            alpha = np.asarray(mask.resize((1024,1024), Image.Resampling.LANCZOS)).astype(np.float32)[:, :, None] / 255
            canvas = np.round(base * (1-alpha) + np.array([255,0,0]) * alpha).astype(np.uint8)
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
