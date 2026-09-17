from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    bg = np.array([240, 235, 230], dtype=np.uint8)
    # Name, original rectangle, start of falling (seconds).
    dominos = [
        ('START', (75,441,114,541), .12),
        ('T1', (197,441,236,541), .46),
        ('T2', (319,441,358,541), .80),
        ('T3', (441,441,480,541), 1.14),
        ('A1', (563,333,602,433), 1.52),
        ('B1', (563,549,602,649), 1.56),
        ('A2', (685,315,724,415), 1.94),
        ('B2', (685,566,724,666), 2.00),
        ('A3', (807,297,846,397), 2.38),
        ('A4', (929,280,968,380), 2.82),
    ]
    clean = source.copy()
    sprites = []
    for name, (x0,y0,x1,y1), start in dominos:
        left = x0-18 if name == 'START' else x0-1
        right = x1+18 if name == 'START' else x1+2
        top, bottom = y0-1, y1+2
        patch = source[top:bottom,left:right].copy()
        mask = np.any(patch != bg, axis=2).astype(np.uint8)*255
        clean[top:bottom,left:right] = bg
        sprites.append((patch,mask,left,top,x0,y1,start))
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    pipe = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
        '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16',
        '-i','-','-an','-c:v','libx264','-crf','16','-preset','medium',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    for frame in range(62):
        t = frame/16
        canvas = clean.copy()
        for patch, mask, left, top, px, py, start in sprites:
            u = np.clip((t-start)/.65,0,1)
            # Accelerating descent with a gentle settling at the floor.
            progress = u*u*(3-2*u)
            angle = math.radians(82*progress)
            # Slight foreshortening keeps the final tile within the original view.
            length = 1-.12*progress
            c,s = math.cos(angle),math.sin(angle)
            a = np.array([[c,-s*length],[s,c*length]],dtype=float)
            offset = np.array([px,py]) + a @ np.array([left-px,top-py])
            matrix = np.column_stack((a,offset))
            if u == 0:
                h,w = mask.shape
                region = canvas[top:top+h,left:left+w]
                region[mask>0] = patch[mask>0]
            else:
                rgba = np.dstack((patch,mask))
                warped = cv2.warpAffine(rgba,matrix,(1024,1024),flags=cv2.INTER_LINEAR)
                alpha = warped[:,:,3:4].astype(float)/255
                # Unpremultiplied texture has a background margin for clean edges.
                canvas = np.rint(canvas*(1-alpha)+warped[:,:,:3]*alpha).astype(np.uint8)
        if frame == 0:
            canvas = source
        pipe.stdin.write(canvas.tobytes())
    pipe.stdin.close()
    if pipe.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
