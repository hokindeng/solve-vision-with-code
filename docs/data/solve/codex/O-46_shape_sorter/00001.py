"""Render the shape sorter using only pixels and masks from the supplied frame."""
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    colors = [(251,146,60), (244,114,182), (250,204,21)]
    destinations = [(757.5,293.5), (757.5,511.5), (757,730)]
    intervals = [(3,25), (28,50), (53,75)]
    cards = []
    for color, destination, interval in zip(colors, destinations, intervals):
        mask = np.all(original == color, axis=2)
        ys, xs = np.where(mask)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()-x+1), int(ys.max()-y+1)
        sprite = mask[y:y+h,x:x+w].astype(np.float32)
        background[mask] = (248,250,252)
        center = (x+(w-1)/2,y+(h-1)/2)
        cards.append((sprite, np.array(color,dtype=np.float32), x,y,
                      destination[0]-center[0],destination[1]-center[1],interval))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
               '-an','-c:v','libx264','-preset','slow','-crf','10',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(output/'video.mp4')]
    encoder = subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame_index in range(78):
        frame = background.copy()
        for sprite,color,x,y,dx,dy,(start,end) in cards:
            t = np.clip((frame_index-start)/(end-start),0,1)
            progress = t*t*(3-2*t)
            px, py = x+dx*progress, y+dy*progress
            ix, iy = int(np.floor(px))-1,int(np.floor(py))-1
            h,w = sprite.shape
            alpha = cv2.warpAffine(sprite,np.float32([[1,0,px-ix],[0,1,py-iy]]),
                                   (w+3,h+3),flags=cv2.INTER_LINEAR)[:,:,None]
            region = frame[iy:iy+h+3,ix:ix+w+3]
            region[:] = np.rint(region*(1-alpha)+color*alpha).astype(np.uint8)
        if frame_index == 0:
            assert np.array_equal(frame,original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
