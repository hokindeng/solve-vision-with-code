from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    w, h = base.size
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUT / 'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    angle = math.radians(-20)
    for frame in range(48):
        image = base.copy()
        if frame > 0:
            progress = min(frame / 44, 1.0)
            overlay = Image.new('RGBA', (w*scale, h*scale), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            points = []
            for t in np.linspace(-math.pi/2, -math.pi/2 + 2*math.pi*progress,
                                 max(2, int(600*progress))):
                u, v = 105*math.cos(t), 50*math.sin(t)
                x = 397 + u*math.cos(angle) - v*math.sin(angle)
                y = 197 + u*math.sin(angle) + v*math.cos(angle)
                points.append((x*scale, y*scale))
            draw.line(points, fill=(230, 25, 30, 255), width=4*scale, joint='curve')
            for x, y in (points[0], points[-1]):
                r = 2*scale
                draw.ellipse((x-r,y-r,x+r,y+r), fill=(230,25,30,255))
            overlay = overlay.resize((w,h), Image.Resampling.LANCZOS)
            image.paste(overlay, (0,0), overlay)
        process.stdin.write(np.asarray(image).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
