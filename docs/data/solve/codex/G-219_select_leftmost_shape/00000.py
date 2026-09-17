from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    w, h = base.size
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # The square spans x=248..396, y=761..909. Leave clearance
    # between its corners and the enclosing circular annotation.
    cx, cy, radius = 322, 835, 119
    scale = 4
    for index in range(48):
        frame = base.copy()
        if index:
            overlay = Image.new('RGBA', (w*scale, h*scale))
            draw = ImageDraw.Draw(overlay)
            progress = index / 47
            box = tuple(v*scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90 + 360*progress, fill=(230, 25, 35, 255), width=5*scale)
            overlay = overlay.resize((w,h), Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
