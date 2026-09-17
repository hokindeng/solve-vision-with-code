from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    bg = original[0, 0]
    foreground = (np.any(original != bg, axis=2)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    objects = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100]
    objects.sort(key=lambda i: (centers[i][1], centers[i][0]))
    borders = []
    for i in objects:
        mask = (labels == i).astype(np.uint8)
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
        borders.append((mask - inner).astype(bool))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '12', '-preset', 'slow', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output / 'video.mp4')], stdin=subprocess.PIPE)
    for frame_number in range(35):
        frame = original.copy()
        # Retained highlighted borders mark objects that have already been counted.
        for j, border in enumerate(borders):
            if frame_number >= 4 + j * 8:
                frame[border] = (0, 115, 230)
        if frame_number >= 28:
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            text = f'Count: {len(objects)}'
            box = draw.textbbox((0, 0), text, font=font)
            x = (1024 - (box[2] - box[0])) / 2 - box[0]
            y = (1024 - (box[3] - box[1])) / 2 - box[1]
            draw.text((x, y), text, fill=(0, 0, 0), font=font)
            frame = np.array(im)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
