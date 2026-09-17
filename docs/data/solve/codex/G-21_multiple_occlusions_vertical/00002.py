from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask_pixels = np.all(original == (209, 209, 209), axis=2)
    ys, xs = np.where(mask_pixels)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mask = original[y0:y1, x0:x1].copy()
    background = original.copy()
    background[y0:y1, x0:x1] = (255, 255, 255)
    height, width = original.shape[:2]
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(58):
        frame = background.copy()
        top = round(y0 + (height - y0) * i / 57)
        bottom = min(height, top + mask.shape[0])
        if top < height:
            frame[top:bottom, x0:x1] = mask[:bottom-top]
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
