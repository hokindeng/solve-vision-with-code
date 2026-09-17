from PIL import Image, ImageDraw
import numpy as np
import subprocess
from pathlib import Path

ROOT = Path('/app')
BASE = Image.open(ROOT/'first_frame.png').convert('RGB')
GREEN = np.array([70,153,53])
PURPLE = np.array([91,30,153])

def ease(t):
    t = max(0., min(1., t))
    return t*t*(3-2*t)

def clear_question(im, x):
    ImageDraw.Draw(im).rectangle((x-24,650,x+24,711), fill='white')

def bar(im, x, y, color):
    d = ImageDraw.Draw(im)
    d.rectangle((x,y,x+160,y+40),fill=(0,0,0))
    d.rectangle((x+2,y+2,x+158,y+38),fill=tuple(int(v) for v in color))

def frame(i):
    im = BASE.copy()
    # Reveal the first answer, then demonstrate its color transformation.
    if i >= 6:
        answered = BASE.copy()
        clear_question(answered,482)
        color = np.rint(GREEN + (PURPLE-GREEN)*ease((i-13)/15))
        bar(answered,402,662,color)
        im = Image.blend(im,answered,ease((i-6)/7))
    # Reveal a copy in the third slot and translate it downward by 100 pixels.
    if i >= 33:
        answered = im.copy()
        clear_question(answered,769)
        y = 662 + round(100*ease((i-39)/15))
        bar(answered,689,y,PURPLE)
        im = Image.blend(im,answered,ease((i-33)/6))
    return im

def main():
    (ROOT/'output').mkdir(exist_ok=True)
    process = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
        '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16',
        '-i','-','-an','-c:v','libx264','-crf','0','-preset','medium',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')],stdin=subprocess.PIPE)
    for i in range(60):
        process.stdin.write(frame(i).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
