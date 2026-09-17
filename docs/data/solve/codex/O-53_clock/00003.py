from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    red = np.all(original == (220, 20, 60), axis=2).astype(np.uint8)
    green = np.all(original == (0, 100, 0), axis=2).astype(np.uint8)
    fixed_black = np.all(original == (0, 0, 0), axis=2)
    base = original.copy()
    base[(red | green).astype(bool)] = (240, 248, 255)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(120):
        if i == 0:
            frame = original.copy()
        else:
            t = i / 119
            # Smooth acceleration and deceleration, with motion throughout.
            progress = t * t * (3 - 2 * t)
            elapsed_hours = 21 * progress
            frame = base.copy()
            for mask, degrees, color in (
                (green, elapsed_hours * 30, (0, 100, 0)),
                (red, elapsed_hours * 360, (220, 20, 60)),
            ):
                matrix = cv2.getRotationMatrix2D((512, 512), -(degrees % 360), 1)
                rotated = cv2.warpAffine(mask, matrix, (1024, 1024), flags=cv2.INTER_NEAREST)
                frame[rotated != 0] = color
            frame[fixed_black] = original[fixed_black]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
