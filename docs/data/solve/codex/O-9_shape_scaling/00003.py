from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def ease(t):
    t = max(0., min(1., t))
    return t*t*(3-2*t)

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # The example's outer T spans 150 units, then 125: scale = 5/6.
    factor = 125 / 150
    # Restrict all drawing to the missing-answer area.
    box = (690, 686, 850, 851)
    initial = original.crop(box)
    white = Image.new('RGB', initial.size, 'white')
    # Sample the source arrow's solid fill.
    pink = original.getpixel((220, 760))
    writer = imageio.get_writer(OUT / 'video.mp4', fps=16,
        codec='libx264', pixelformat='yuv420p', macro_block_size=1,
        ffmpeg_params=['-crf', '18', '-preset', 'slow'])
    for i in range(60):
        frame = original.copy()
        if i:
            patch = Image.blend(initial, white, ease(i / 15))
            if i >= 15:
                # Introduce the same right-facing arrow, then smoothly scale it
                # around its bounding-box center to match the example.
                progress = ease((i - 15) / 39)
                scale = 1 + (factor - 1) * progress
                arrow = white.copy()
                points = [(-75,-37.5),(0,-37.5),(0,-75),
                          (75,0),(0,75),(0,37.5),(-75,37.5)]
                points = [(round(769 + x*scale-box[0]),
                           round(768 + y*scale-box[1])) for x,y in points]
                ImageDraw.Draw(arrow).polygon(points, fill=pink, outline='black')
                patch = Image.blend(white, arrow, ease((i-15)/10))
            frame.paste(patch, box[:2])
        writer.append_data(np.asarray(frame))
    writer.close()

if __name__ == '__main__':
    main()
