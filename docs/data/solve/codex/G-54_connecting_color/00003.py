from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def bezier(points):
    p = np.array(points, dtype=float)
    t = np.linspace(0, 1, 1200)[:, None]
    q = (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    return q, lengths / lengths[-1]

def main():
    (ROOT/'output').mkdir(exist_ok=True)
    original = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    blank = np.all(original == 255, axis=2)
    curves = [
        ([(444,285),(564,335),(725,337),(841,276)], (170,178,255)),
        ([(178,528),(279,425),(389,432),(487,567)], (72,219,251)),
        ([(357,796),(440,883),(540,890),(625,793)], (29,209,161)),
    ]
    paths = [(bezier(p), c) for p,c in curves]
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')], stdin=subprocess.PIPE)
    for frame in range(48):
        result = original.copy()
        for k,((q,lengths),color) in enumerate(paths):
            progress = np.clip((frame - k*15)/15, 0, 1)
            if progress <= 0:
                continue
            # Arc-length progression makes the drawing speed even along the curve.
            end = np.searchsorted(lengths, progress, side='right')
            pts = q[:max(2,end)]
            mask = Image.new('L',(4096,4096))
            draw = ImageDraw.Draw(mask)
            coords = [(round(x*4),round(y*4)) for x,y in pts]
            draw.line(coords,fill=255,width=24)
            for x,y in (coords[0],coords[-1]):
                draw.ellipse((x-12,y-12,x+12,y+12),fill=255)
            alpha = np.asarray(mask.resize((1024,1024),Image.Resampling.LANCZOS),dtype=float)/255
            alpha *= blank
            result = np.rint(result*(1-alpha[:,:,None])+np.array(color)*alpha[:,:,None]).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
