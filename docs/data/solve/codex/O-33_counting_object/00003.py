from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(base != base[0, 0], axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    objects = sorted(range(1, count), key=lambda n: (centers[n][1], centers[n][0]))
    black = np.all(base == 0, axis=2)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '0', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output / 'video.mp4')], stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    for frame in range(30):
        pixels = base.copy()
        if 4 <= frame < 22:
            index = min((frame - 4) // 9, len(objects) - 1)
            pixels[(labels == objects[index]) & black] = (255, 180, 0)
        image = Image.fromarray(pixels)
        if frame >= 22:
            draw = ImageDraw.Draw(image)
            text = f'Count: {len(objects)}'
            box = draw.textbbox((0, 0), text, font=font)
            x = (1024 - (box[2] - box[0])) / 2 - box[0]
            y = (1024 - (box[3] - box[1])) / 2 - box[1]
            draw.text((x, y), text, font=font, fill=(0, 0, 0))
        proc.stdin.write(np.asarray(image).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
