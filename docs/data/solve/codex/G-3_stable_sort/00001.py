from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def ease(t):
    t = np.clip(t, 0, 1)
    return t*t*t*(10 + t*(-15 + 6*t))

def main():
    original = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    bg = original[0, 0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.any(original != bg, axis=2).astype(np.uint8), 8)
    shapes = {}
    for label in range(1, count):
        x,y,w,h,_ = stats[label]
        shapes[label] = dict(pixels=original[y:y+h,x:x+w].copy(),
                             mask=labels[y:y+h,x:x+w] == label,
                             start=np.array([x,y], dtype=float), w=w, h=h)
    # First gather each type in its own column, preserving its initial order.
    grouped_centers = {1:(250,240),4:(250,510),6:(250,780),
                       2:(770,240),3:(770,510),5:(770,780)}
    order = [4,6,1,2,3,5]
    gap = 22
    left = (1024 - sum(shapes[i]['w'] for i in order) - gap*5)//2
    for i in order:
        s = shapes[i]
        s['group'] = np.array(grouped_centers[i])-np.array([(s['w']-1)/2,(s['h']-1)/2])
        s['end'] = np.array([left,512-(s['h']-1)/2])
        left += s['w']+gap
    (ROOT/'output').mkdir(exist_ok=True)
    encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024',
        '-r','16','-i','-','-an','-c:v','libx264','-crf','16',
        '-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',
        str(ROOT/'output/video.mp4')], stdin=subprocess.PIPE)
    for f in range(96):
        canvas = np.full_like(original, bg)
        for i,s in shapes.items():
            if f <= 43:
                t = ease((f-3)/40)
                pos = s['start']*(1-t)+s['group']*t
            else:
                t = ease((f-46)/46)
                pos = s['group']*(1-t)+s['end']*t
            x,y = np.rint(pos).astype(int)
            region = canvas[y:y+s['h'],x:x+s['w']]
            region[s['mask']] = s['pixels'][s['mask']]
        if f == 0:
            assert np.array_equal(canvas, original)
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
