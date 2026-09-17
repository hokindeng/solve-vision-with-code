from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(base)
    # Find the bounds of the central, innermost solid-color square.
    color = pixels[base.height // 2, base.width // 2]
    ys, xs = np.where(np.all(pixels == color, axis=2))
    left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    width = 6
    blue = (0, 0, 255)
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)
    ], stdin=subprocess.PIPE)
    for frame_number in range(85):
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        # Four sequential strokes: top, right, bottom, then left.
        # Each side has 16 drawing frames followed by a two-frame pause.
        for side in range(4):
            progress = min(1.0, max(0.0, (frame_number - 4 - side * 18) / 16))
            if progress <= 0:
                continue
            length = max(1, round((right - left + 1) * progress))
            if side == 0:
                box = (left, top, left + length - 1, top + width - 1)
            elif side == 1:
                box = (right - width + 1, top, right, top + length - 1)
            elif side == 2:
                box = (right - length + 1, bottom - width + 1, right, bottom)
            else:
                box = (left, bottom - length + 1, left + width - 1, bottom)
            draw.rectangle(box, fill=blue)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
