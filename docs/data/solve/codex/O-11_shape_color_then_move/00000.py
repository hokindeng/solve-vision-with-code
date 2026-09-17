from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def ease(t):
    t = max(0., min(1., t))
    return t*t*(3-2*t)

def main():
    original = Image.open(ROOT/'first_frame.png').convert('RGB')
    source = original.crop((83,626,308,739))
    sprite = np.array(source)
    green = np.array([7,153,80])
    cyan = np.array([45,229,168])
    fill = np.all(sprite == cyan, axis=2)
    out = ROOT/'output/video.mp4'
    out.parent.mkdir(exist_ok=True)
    process = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',str(out)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for n in range(60):
        frame = original.copy()
        # Reveal the middle term, then demonstrate its color change.
        if n > 5:
            alpha = ease((n-5)/8)
            color_t = ease((n-13)/15)
            middle = sprite.copy()
            middle[fill] = np.rint(cyan*(1-color_t)+green*color_t).astype(np.uint8)
            target = original.copy()
            ImageDraw.Draw(target).rectangle((369,626,594,739), fill='white')
            target.paste(Image.fromarray(middle), (370,626))
            frame = Image.blend(frame,target,alpha)
        # Carry the recolored rectangle into the final term and lower it
        # by the same 25 pixels demonstrated by the circles above.
        if n > 31:
            alpha = ease((n-31)/7)
            offset = round(25*ease((n-38)/15))
            final_sprite = sprite.copy()
            final_sprite[fill] = green
            target = frame.copy()
            ImageDraw.Draw(target).rectangle((656,626,882,764), fill='white')
            target.paste(Image.fromarray(final_sprite),(657,626+offset))
            frame = Image.blend(frame,target,alpha)
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
