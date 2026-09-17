from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')
def main():
    original = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Retain the original line drawing, including the arrow and callout.
    x0,y0,x1,y1 = 117,351,205,462
    crop = original[y0:y1,x0:x1].copy()
    green = (crop[:,:,0] == 0) & (crop[:,:,1] > 0)
    outline = np.all(crop == (50,50,50),axis=2)
    mask = green | outline
    base = original.copy()
    base[y0:y1,x0:x1][mask] = 255
    # Recover the small strip of the right face hidden by the arrow.
    for y in range(418,425):
        for x in range(160,205):
            p = original[y,x]
            if np.array_equal(p,[255,255,255]) or np.array_equal(p,[255,0,0]):
                crop[y-y0,x-x0] = (50,50,50) if x in (160,204) else (0,124,60)
                mask[y-y0,x-x0] = True
    sprite = np.zeros((1024,1024,4),np.uint8)
    sprite[y0:y1,x0:x1,:3] = crop
    sprite[y0:y1,x0:x1,3] = mask.astype(np.uint8)*255
    # Premultiplication avoids fringes during fractional-pixel motion.
    rgba = sprite.astype(np.float32)/255
    rgba[:,:,:3] *= rgba[:,:,3:4]
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
         '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
         '-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i in range(46):
        if i == 0:
            frame=original
        else:
            t=min(i/42,1.0)
            # Smooth acceleration, travel along the indicated downward arc,
            # and a gentle deceleration into the outlined socket.
            u=t*t*(3-2*t)
            dx=621*u
            dy=538*u*u
            moved=cv2.warpAffine(rgba,np.float32([[1,0,dx],[0,1,dy]]),(1024,1024),flags=cv2.INTER_LINEAR)
            frame=np.clip(moved[:,:,:3]*255+base*(1-moved[:,:,3:4]),0,255).round().astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    err=proc.stderr.read()
    if proc.wait():
        raise RuntimeError(err.decode())

if __name__ == '__main__':
    main()
