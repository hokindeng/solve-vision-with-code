from pathlib import Path
import math
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    sprites = []
    for left, right in [(139, 177), (205, 243)]:
        sprite = np.zeros((1024, 1024, 4), dtype=np.uint8)
        sprite[617:748, left:right+1, :3] = original[617:748, left:right+1]
        sprite[617:748, left:right+1, 3] = 255
        sprites.append(sprite)
        background[617:746, left:right+1] = 255
        background[746:748, left:right+1] = (139, 115, 85)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    writer = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    contact = math.asin(28 / 130)
    for frame in range(50):
        if frame <= 4:
            rendered = original.copy()
        else:
            if frame < 16:
                a = contact * ((frame - 4) / 12) ** 2
                b = 0.0
            else:
                t = min(1.0, (frame - 16) / 27)
                b = math.pi / 2 * t ** 1.65
                # The leading upper corner of 1 stays against the face of 2.
                a = b + math.asin((66 * math.cos(b) - 38) / 130)
            rendered = background.copy()
            for i, angle in [(1, b), (0, a)]:
                pivot = (243, 747) if i else (177, 747)
                matrix = cv2.getRotationMatrix2D(pivot, -math.degrees(angle), 1)
                layer = cv2.warpAffine(sprites[i], matrix, (1024, 1024), flags=cv2.INTER_CUBIC)
                alpha = layer[:, :, 3:4].astype(float) / 255
                # Unpremultiply the interpolated sprite border before compositing.
                rgb = layer[:, :, :3].astype(float)
                rendered = np.clip(rgb + rendered * (1 - alpha), 0, 255).astype(np.uint8)
        writer.stdin.write(rendered.tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
