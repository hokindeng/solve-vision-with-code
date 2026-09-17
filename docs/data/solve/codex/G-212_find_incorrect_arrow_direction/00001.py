from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    width, height = base.size
    # The lower arrow runs opposite to the other three around the circle.
    center_x, center_y, radius = 402, 769, 103
    scale = 4
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(OUT / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(48):
        frame = base.copy()
        if index > 0:
            progress = min(index / 45.0, 1.0)
            overlay = Image.new('RGBA', (width * scale, height * scale))
            draw = ImageDraw.Draw(overlay)
            bounds = tuple(v * scale for v in (
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius))
            draw.arc(bounds, start=-90, end=-90 + 360 * progress,
                     fill=(235, 25, 30, 255), width=4 * scale)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0, 0), overlay)
        encoder.stdin.write(np.asarray(frame).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
