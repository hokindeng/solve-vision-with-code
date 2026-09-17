from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    colors = [(244,114,182), (248,113,113), (250,204,21)]
    moves = [(566,-48), (423,162), (562,45)]
    masks = [np.all(original == color, axis=2).astype(np.float32) for color in colors]
    background = original.copy()
    for mask in masks:
        background[mask.astype(bool)] = (248,250,252)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
               '-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
               '-movflags','+faststart', str(OUT)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(78):
        canvas = background.copy()
        for i, (mask, color, (dx,dy)) in enumerate(zip(masks,colors,moves)):
            start, end = [(3,25),(27,49),(51,74)][i]
            t = np.clip((frame-start)/(end-start), 0., 1.)
            t = t*t*(3.-2.*t)
            shifted = cv2.warpAffine(mask,np.float32([[1,0,dx*t],[0,1,dy*t]]),
                                     (1024,1024), flags=cv2.INTER_LINEAR)
            visible = shifted > 0
            alpha = shifted[visible,None]
            canvas[visible] = np.rint(canvas[visible]*(1-alpha)+np.array(color)*alpha).astype(np.uint8)
        if frame == 0:
            assert np.array_equal(canvas,original)
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
