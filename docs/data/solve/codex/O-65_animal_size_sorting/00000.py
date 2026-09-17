from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Keep the original raster artwork, including its fine outlines.
    boxes = [(277,533,420,697), (639,331,765,435), (109,60,209,155)]
    centers = [332,512,692]  # descending size: fox, frog, panda
    background = original.copy()
    sprites = []
    for box, center in zip(boxes, centers):
        x0,y0,x1,y1 = box
        crop = original[y0:y1,x0:x1].copy()
        mask = np.any(crop != 255, axis=2)
        background[y0:y1,x0:x1][mask] = 255
        sprites.append((crop,mask,(x0,y0),(center-(x1-x0)//2,944-(y1-y0))))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    with imageio.get_writer(out / 'video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', quality=10, macro_block_size=1, ffmpeg_params=['-crf','0']) as writer:
        for i in range(40):
            t = i / 39
            ease = t*t*(3-2*t)
            frame = background.copy()
            for crop,mask,start,end in sprites:
                x,y = [round(a+(b-a)*ease) for a,b in zip(start,end)]
                h,w = mask.shape
                frame[y:y+h,x:x+w][mask] = crop[mask]
            if i == 0:
                assert np.array_equal(frame, original)
            writer.append_data(frame)

if __name__ == '__main__':
    main()
