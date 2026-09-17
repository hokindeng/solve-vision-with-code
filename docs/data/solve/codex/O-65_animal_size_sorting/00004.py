from pathlib import Path
import math
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Original bounds, ordered by the size of each animal face.
    bounds = [(364,120,190,118), (287,254,117,129),
              (743,387,102,98), (352,712,68,101), (624,673,67,55)]
    background = original.copy()
    sprites = []
    for x,y,w,h in bounds:
        pixels = original[y:y+h,x:x+w].copy()
        mask = np.any(pixels != 255, axis=2)
        sprites.append((pixels,mask))
        background[y:y+h,x:x+w][mask] = 255
    baseline = 944
    gap = 65
    left = (1024 - sum(b[2] for b in bounds) - gap*4)//2
    destinations = []
    for x,y,w,h in bounds:
        destinations.append((left,baseline-h))
        left += w+gap
    (ROOT/'output').mkdir(exist_ok=True)
    with imageio.get_writer(ROOT/'output/video.mp4', fps=16, codec='libx264',
                            pixelformat='yuv420p', quality=10,
                            macro_block_size=None, ffmpeg_params=['-crf','0']) as writer:
        for frame in range(40):
            t = frame/39
            u = t*t*(3-2*t)
            canvas = background.copy()
            for i, ((x,y,w,h),(tx,ty),(pixels,mask)) in enumerate(zip(bounds,destinations,sprites)):
                # A shallow leftward arc keeps the dog clear of the cat.
                arc = -100*math.sin(math.pi*u) if i == 0 else 0
                nx = round(x+(tx-x)*u+arc)
                ny = round(y+(ty-y)*u)
                region = canvas[ny:ny+h,nx:nx+w]
                region[mask] = pixels[mask]
            if frame == 0:
                assert np.array_equal(canvas,original)
            writer.append_data(canvas)

if __name__ == '__main__':
    main()
