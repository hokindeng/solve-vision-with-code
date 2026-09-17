from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    width, height = base.size
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # The scalene purple triangle is the only shape without reflection symmetry.
    # The mark surrounds it without touching its original pixels.
    scale = 4
    cx, cy, radius = 379, 528, 77
    box = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
    for i in range(16):
        frame = base.copy()
        if i:
            overlay = Image.new('RGBA', (width*scale, height*scale))
            draw = ImageDraw.Draw(overlay)
            draw.arc(box, start=-90, end=-90+360*i/15, fill=(255, 0, 0, 255), width=4*scale)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0, 0), overlay)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
