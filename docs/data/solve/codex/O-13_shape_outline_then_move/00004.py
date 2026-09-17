from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'

def ease(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3-2*t)

def main():
    OUT.parent.mkdir(exist_ok=True)
    original = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Copy the actual minus, including its precise outline and dimensions.
    minus = original[661:704,107:270].copy()
    blue = np.array([30,91,153], dtype=float)
    p = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
        '-an','-c:v','libx264','-crf','0','-preset','slow',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)],stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    for i in range(64):
        frame = original.copy()
        if i:
            # Reveal the first answer, then remove its fill to demonstrate step 1.
            reveal = ease((i-5)/10)
            box = frame[658:706,459:494].astype(float)
            frame[658:706,459:494] = np.rint(box*(1-reveal)+255*reveal).astype('uint8')
            shape = minus.copy()
            fill = 1-ease((i-15)/15)
            shape[3:40,2:161] = np.rint(255*(1-fill)+blue*fill).astype('uint8')
            x,y = 395,661
            area = frame[y:y+43,x:x+163].astype(float)
            frame[y:y+43,x:x+163] = np.rint(area*(1-reveal)+shape*reveal).astype('uint8')
            # Repeat the outline in the final column, then translate down by
            # the same 60 pixels illustrated by the top-row square.
            reveal2 = ease((i-32)/9)
            box = frame[658:706,747:782].astype(float)
            frame[658:706,747:782] = np.rint(box*(1-reveal2)+255*reveal2).astype('uint8')
            y = 661+round(60*ease((i-41)/17))
            x = 683
            area = frame[y:y+43,x:x+163].astype(float)
            frame[y:y+43,x:x+163] = np.rint(area*(1-reveal2)+minus*reveal2).astype('uint8')
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
