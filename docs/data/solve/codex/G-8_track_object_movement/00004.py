from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The isolated selection contains the hexagon and its green tracking border.
    x0, y0, x1, y1 = 250, 380, 452, 612
    sprite = original[y0:y1, x0:x1].copy()
    mask = np.any(sprite != 255, axis=2)
    background = original.copy()
    patch = background[y0:y1, x0:x1]
    patch[mask] = 255
    # Align the centers using the star and the tracking border's bounding boxes.
    red = (original[:, :, 0] == 255) & (original[:, :, 1] == 0) & (original[:, :, 2] == 0)
    star_x = np.where(red)[1]
    target = (star_x.min() + star_x.max()) / 2
    sprite_x = np.where(mask)[1]
    center = x0 + (sprite_x.min() + sprite_x.max()) / 2
    distance = int(round(target - center))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(60):
        t = i / 59
        ease = t * t * (3 - 2 * t)
        dx = round(distance * ease)
        frame = background.copy()
        region = frame[y0:y1, x0+dx:x1+dx]
        region[mask] = sprite[mask]
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    errors = process.stderr.read()
    if process.wait():
        raise RuntimeError(errors.decode())
    print(f'Created {out / "video.mp4"}; horizontal displacement: {distance}px')

if __name__ == '__main__':
    main()
