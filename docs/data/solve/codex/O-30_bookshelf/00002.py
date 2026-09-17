from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def position(start, target, step):
    # Lift clear of the tallest book, carry across, then lower into the gap.
    sx, sy = start
    tx, ty = target
    top = 24
    if step <= 3:
        t = step / 3
        return sx, round(sy + (top-sy)*t)
    if step <= 8:
        t = (step-3)/5
        return round(sx + (tx-sx)*t), top
    t = min(1, (step-8)/3)
    return tx, round(top + (ty-top)*t)

def main():
    original = Image.open(ROOT/'first_frame.png').convert('RGB')
    books = [(736,308,768,513), (773,278,805,513)]
    sprites = [original.crop(box) for box in books]
    background = np.array(original).copy()
    for x0,y0,x1,y1 in books:
        background[y0:512,x0:x1] = (255,255,255)
        background[512:y1,x0:x1] = (52,41,22)
    destinations = [(334,308), (556,278)]
    output = ROOT/'output/video.mp4'
    output.parent.mkdir(exist_ok=True)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    process = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for frame in range(24):
        canvas = Image.fromarray(background)
        p0 = position((736,308),destinations[0],min(frame,11))
        p1 = position((773,278),destinations[1],max(0,min(frame-11,11)))
        canvas.paste(sprites[0],p0)
        canvas.paste(sprites[1],p1)
        if frame == 0:
            assert np.array_equal(np.array(canvas),np.array(original))
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
