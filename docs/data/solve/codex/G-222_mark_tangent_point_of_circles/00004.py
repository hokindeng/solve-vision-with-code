from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    (ROOT / 'output').mkdir(exist_ok=True)
    scale = 4
    # The two large circles meet at (511.5, 380.5).
    cx, cy, radius = 511.5, 380.5, 23
    writer = imageio.get_writer(str(ROOT / 'output/video.mp4'), fps=16,
                               codec='libx264', pixelformat='yuv420p',
                               quality=10, macro_block_size=1)
    try:
        for frame in range(60):
            im = base.copy()
            progress = min(1.0, max(0.0, (frame - 4) / 51))
            if progress > 0:
                mask = Image.new('L', (1024*scale, 1024*scale), 0)
                draw = ImageDraw.Draw(mask)
                points = []
                for angle in np.linspace(-math.pi/2, -math.pi/2 + progress*2*math.pi,
                                         max(2, int(progress*360)+1)):
                    points.append(((cx+radius*math.cos(angle))*scale,
                                   (cy+radius*math.sin(angle))*scale))
                draw.line(points, fill=255, width=4*scale, joint='curve')
                for x,y in (points[0], points[-1]):
                    draw.ellipse((x-2*scale,y-2*scale,x+2*scale,y+2*scale),fill=255)
                mask = mask.resize(base.size, Image.Resampling.LANCZOS)
                im.paste((0,0,0), (0,0), mask)
            writer.append_data(np.asarray(im))
    finally:
        writer.close()

if __name__ == '__main__':
    main()
