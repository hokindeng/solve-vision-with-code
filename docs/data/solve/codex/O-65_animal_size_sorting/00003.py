from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # Tight bounds of each original animal, ordered by its visible size.
    bounds = [(100,146,264,280), (585,422,704,559),
              (214,615,309,706), (857,146,941,227),
              (931,526,986,608)]
    centers = [160,350,540,730,920]
    background = original.copy()
    sprites = []
    for box, center in zip(bounds, centers):
        sprite = original.crop(box)
        background.paste((255,255,255), box)
        x,y,r,b = box
        sprites.append((sprite, x,y,center-sprite.width//2,944-sprite.height))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    with imageio.get_writer(out / 'video.mp4', format='FFMPEG', mode='I',
                            fps=16, codec='libx264', pixelformat='yuv420p',
                            macro_block_size=1, ffmpeg_params=['-crf','18']) as writer:
        for i in range(40):
            t = i/39
            u = t*t*(3-2*t)
            frame = background.copy()
            for sprite,x,y,tx,ty in sprites:
                frame.paste(sprite,(round(x+(tx-x)*u),round(y+(ty-y)*u)))
            if i == 0:
                assert np.array_equal(np.asarray(frame),np.asarray(original))
            writer.append_data(np.asarray(frame))

if __name__ == '__main__':
    main()
