from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    bg = base[0, 0]
    foreground = np.any(base != bg, axis=2).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    objects = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 100]
    objects.sort(key=lambda i: (centers[i][1], centers[i][0]))
    borders = []
    for i in objects:
        mask = (labels == i).astype(np.uint8)
        inner = cv2.erode(mask, np.ones((7, 7), np.uint8))
        borders.append((mask - inner).astype(bool))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 52)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_number in range(40):
        frame = base.copy()
        # Four distinct counting intervals, in top-to-bottom order.
        if 2 <= frame_number < 34:
            index = (frame_number - 2) // 8
            frame[borders[index]] = (255, 200, 0)
        if frame_number >= 34:
            canvas = Image.fromarray(frame)
            draw = ImageDraw.Draw(canvas)
            text = f'Count: {len(objects)}'
            box = draw.textbbox((0, 0), text, font=font)
            x = (1024 - (box[2] - box[0])) / 2 - box[0]
            y = (1024 - (box[3] - box[1])) / 2 - box[1]
            draw.text((x, y), text, fill=(0, 0, 0), font=font)
            frame = np.array(canvas)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    error = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(error.decode())
    print(f'Created {out / "video.mp4"}: 40 frames, {len(objects)} objects')

if __name__ == '__main__':
    main()
