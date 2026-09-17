from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    black = np.all(original == (0, 0, 0), axis=2)
    yy, xx = np.where(black)
    start = np.array([(xx.min()+xx.max())/2, (yy.min()+yy.max())/2])
    radius = (xx.max()-xx.min()+1)/2
    colors = [(255,140,0), (255,105,180), (60,179,113)]
    masks = [np.all(original == c, axis=2) for c in colors]
    base = original.copy()
    base[black] = 255
    # Each meal makes the eater large enough for the next ball.
    stages = [(6,32,np.array([97.,697.]),44.),
              (40,64,np.array([917.,309.]),66.),
              (71,89,np.array([930.,655.]),90.)]
    Y,X = np.ogrid[:1024,:1024]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
        '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16',
        '-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(95):
        canvas = base.copy()
        pos, r = start.copy(), radius
        for i,(begin,end,target,new_r) in enumerate(stages):
            if frame < begin:
                break
            if frame >= end:
                canvas[masks[i]] = 255
                pos,r = target.copy(),new_r
                continue
            t=(frame-begin)/(end-begin)
            ease=t*t*(3-2*t)
            pos=pos+(target-pos)*ease
            g=np.clip((t-.70)/.30,0,1)
            g=g*g*(3-2*g)
            r=r+(new_r-r)*g
            break
        canvas[(X-pos[0])**2+(Y-pos[1])**2 <= r*r] = 0
        if frame <= 5:
            canvas=original.copy()
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
