from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    # Sprite rectangles include the START label that extends beyond its domino.
    specs = [
        ((57,456,130,583),(119,582),2),
        ((211,456,264,583),(263,582),10),
        ((355,363,408,490),(407,489),18),
        ((355,549,408,676),(407,675),18),
        ((499,338,552,466),(551,465),26),
        ((499,573,552,700),(551,699),26),
        ((643,313,696,440),(695,439),34),
        ((643,598,696,726),(695,725),34),
    ]
    sprites = []
    for (x0,y0,x1,y1),pivot,start in specs:
        patch = original[y0:y1,x0:x1].copy()
        alpha = np.any(patch != [240,240,240],axis=2).astype(np.float32)
        rgba = np.dstack((patch.astype(np.float32)*alpha[:,:,None],alpha))
        sprites.append((rgba,(x0,y0),pivot,start))
        background[y0:y1,x0:x1] = (240,240,240)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
        '-an','-c:v','libx264','-crf','16','-preset','medium',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')
    ],stdin=subprocess.PIPE)
    for f in range(50):
        if f == 0:
            frame = original
        else:
            frame = background.astype(np.float32)
            for rgba,(x,y),(px,py),start in sprites:
                t = np.clip((f-start)/11.0,0,1)
                # Accelerate under gravity, then settle into a nearly horizontal tilt.
                ease = t*t*(3-2*t)
                angle = np.deg2rad(79*ease)
                c,s = np.cos(angle),np.sin(angle)
                matrix = np.array([[c,-s,px+c*(x-px)-s*(y-py)],
                                   [s,c,py+s*(x-px)+c*(y-py)]],np.float32)
                moved = cv2.warpAffine(rgba,matrix,(1024,1024),flags=cv2.INTER_LINEAR)
                a = moved[:,:,3:4]
                frame = frame*(1-a)+moved[:,:,:3]
            frame = np.clip(np.rint(frame),0,255).astype(np.uint8)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
