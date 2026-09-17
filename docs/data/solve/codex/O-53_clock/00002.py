from pathlib import Path
import math
import subprocess
import numpy as np
import cv2
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'
BG = (240, 248, 255)
HOUR = (75, 0, 130)
MINUTE = (65, 105, 225)

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    hour_mask = np.all(original == HOUR, axis=2)
    minute_mask = np.all(original == MINUTE, axis=2)
    hub = np.all(original == (0, 0, 0), axis=2)
    face = original.copy()
    face[hour_mask | minute_mask] = BG
    OUT.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(OUT)
    ], stdin=subprocess.PIPE)

    def hand(frame, angle, length, width, color):
        theta = math.radians(angle)
        direction = np.array([math.sin(theta), -math.cos(theta)])
        normal = np.array([-direction[1], direction[0]]) * width / 2
        start = np.array([512., 512.])
        end = start + length * direction
        corners = np.array([start + normal, end + normal, end - normal, start - normal])
        cv2.fillConvexPoly(frame, np.round(corners * 256).astype(np.int32), color, shift=8)

    for index in range(120):
        if index == 0:
            frame = original.copy()
        else:
            t = index / 119
            # Smooth acceleration and deceleration over the entire 16-hour advance.
            progress = t * t * (3 - 2 * t)
            elapsed_minutes = 960 * progress
            frame = face.copy()
            if index == 119:
                # Sixteen whole hours leave the minute hand in exactly its original position.
                frame[minute_mask] = MINUTE
            else:
                hand(frame, 216 + 6 * elapsed_minutes, 286.7, 4, MINUTE)
            hand(frame, 228 + elapsed_minutes / 2, 204.8, 7, HOUR)
            frame[hub] = original[hub]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
