from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path('/app')
def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # Each crop includes the original black outline, down to the shelf top.
    boxes = [(670,382,695,513), (700,287,725,513),
             (730,354,755,513), (760,354,785,513)]
    destinations = [69,519,159,249]
    sprites = [original.crop(b) for b in boxes]
    background = original.copy()
    for b in boxes:
        background.paste((255,255,255), b)
    frames = [np.array(original)]
    placed = {}
    for index, (box, sprite, destination) in enumerate(zip(boxes,sprites,destinations)):
        steps = 6 if index < 3 else 5
        for step in range(1,steps+1):
            t = step / steps
            # Lift clear of the tallest book, translate, and lower into the gap.
            x0,y0 = box[:2]
            lifted_y = 245 - sprite.height
            if t <= .30:
                x = x0
                y = y0 + (lifted_y-y0)*t/.30
            elif t <= .70:
                x = x0 + (destination-x0)*(t-.30)/.40
                y = lifted_y
            else:
                x = destination
                y = lifted_y + (y0-lifted_y)*(t-.70)/.30
            canvas = background.copy()
            for j,s in enumerate(sprites):
                if j == index:
                    continue
                pos = placed.get(j, boxes[j][:2])
                canvas.paste(s, pos)
            canvas.paste(sprite, (round(x),round(y)))
            frames.append(np.asarray(canvas))
        placed[index] = (destination,box[1])
    (ROOT/'output').mkdir(exist_ok=True)
    with imageio.get_writer(ROOT/'output/video.mp4', fps=16, codec='libx264',
                            pixelformat='yuv420p', macro_block_size=1,
                            ffmpeg_params=['-crf','0','-preset','medium']) as writer:
        for frame in frames:
            writer.append_data(frame)

if __name__ == '__main__':
    main()
