from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t*t*(3.0-2.0*t)

def main():
    source = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    yy, xx = np.indices(source.shape[:2])
    # Select only the two colored symbols; position frames and labels stay intact.
    yellow = ((xx >= 680) & (xx <= 762) & (yy >= 470) & (yy <= 552)
              & (source[:,:,0] > 180) & (source[:,:,1] > 180) & (source[:,:,2] < 180))
    teal = ((xx >= 790) & (xx <= 864) & (yy >= 475) & (yy <= 548)
            & (source[:,:,0] < 200) & (source[:,:,1] > source[:,:,0])
            & (source[:,:,2] > source[:,:,0]))
    # A color-difference layer preserves the original symbol's rasterization.
    layer = np.zeros_like(source, dtype=np.float32)
    layer[teal] = source[teal].astype(np.float32)-255
    base = source.copy()
    base[teal] = 255
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    pipe = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for i in range(45):
        frame = base.astype(np.float32)
        fade = smooth(i/21)
        frame[yellow] = source[yellow]*(1-fade)+255*fade
        shift = 105*smooth((i-23)/19)
        moved = cv2.warpAffine(layer, np.float32([[1,0,-shift],[0,1,0]]),
                               (1024,1024), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        frame = np.clip(np.rint(frame+moved),0,255).astype(np.uint8)
        if i == 0:
            frame = source
        pipe.stdin.write(frame.tobytes())
    pipe.stdin.close()
    if pipe.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
