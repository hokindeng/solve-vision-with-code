from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    first = Image.open(ROOT / 'first_frame.png').convert('RGB')
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    circles = [((394, 307), 380, (210, 245, 60)),
               ((847, 716), 163, (240, 50, 230))]
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '10', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(40):
        t = i / 39
        t = t * t * (3 - 2 * t)
        frame = Image.new('RGB', first.size, (255, 255, 255))
        draw = ImageDraw.Draw(frame)
        for (x0, y0), radius, color in circles:
            x = round(x0 + (512 - x0) * t)
            y = round(y0 + (512 - y0) * t)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         outline=color, width=8)
        if i == 0:
            assert np.array_equal(np.asarray(frame), np.asarray(first))
            frame = first
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
