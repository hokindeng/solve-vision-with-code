from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
background = (255, 250, 205)
hour_color = (139, 69, 19)
minute_color = (147, 112, 219)
base = source.copy()
for color in (hour_color, minute_color):
    base[np.all(source == color, axis=2)] = background
pin = np.all(source == (0, 0, 0), axis=2)
# Reconstruct the portions hidden beneath the stationary central pin.
hour = np.zeros((1024, 1024), np.uint8)
minute = np.zeros_like(hour)
hour[512:717, 509:517] = 255
minute[225:513, 510:515] = 255

def rotate(mask, clockwise_degrees):
    matrix = cv2.getRotationMatrix2D((512, 512), -clockwise_degrees, 1)
    return cv2.warpAffine(mask, matrix, (1024, 1024), flags=cv2.INTER_NEAREST) != 0

cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
       '-pixel_format', 'rgb24', '-video_size', '1024x1024', '-framerate', '16',
       '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for index in range(120):
    if index == 0:
        frame = source.copy()
    else:
        t = index / 119
        # Smooth acceleration and deceleration, with motion throughout the clip.
        progress = t * t * (3 - 2 * t)
        hours_elapsed = 15 * progress
        frame = base.copy()
        frame[rotate(minute, hours_elapsed * 360)] = minute_color
        frame[rotate(hour, hours_elapsed * 30)] = hour_color
        frame[pin] = source[pin]
    encoder.stdin.write(frame.tobytes())
encoder.stdin.close()
if encoder.wait():
    raise RuntimeError('Video encoding failed')
