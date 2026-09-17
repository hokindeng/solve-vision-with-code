from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path('/app')
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
bg = base[0, 0]
foreground = np.any(base != bg, axis=2).astype(np.uint8)
count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
objects = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100]
objects.sort(key=lambda i: (stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT]))
borders = []
for i in objects:
    mask = (labels == i).astype(np.uint8)
    interior = cv2.erode(mask, np.ones((7, 7), np.uint8))
    borders.append((mask - interior).astype(bool))
assert len(objects) == 10
out = ROOT / 'output'
out.mkdir(exist_ok=True)
encoder = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', str(out / 'video.mp4')
], stdin=subprocess.PIPE)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 48)
for frame_number in range(70):
    frame = base.copy()
    # Reveal one new highlighted outline every six frames, retaining counted outlines.
    highlighted = min(len(objects), (frame_number + 5) // 6) if frame_number else 0
    for border in borders[:highlighted]:
        frame[border] = (245, 55, 35)
    if frame_number >= 61:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((512, 512), f'Count: {len(objects)}', font=font, fill=(0, 0, 0), anchor='mm')
        frame = np.array(img)
    encoder.stdin.write(frame.tobytes())
encoder.stdin.close()
if encoder.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
