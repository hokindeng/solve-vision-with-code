from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')

def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3-2*t)

def main():
    original = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    bg = original[0,0]
    foreground = np.any(original != bg, axis=2).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    objects = []
    for label in range(1,n):
        x,y,w,h,area = stats[label]
        crop = original[y:y+h,x:x+w].copy()
        mask = labels[y:y+h,x:x+w] == label
        kind = 'square' if np.any(np.all(crop == [230,105,105],axis=2)) else 'triangle'
        objects.append(dict(kind=kind,w=w,h=h,crop=crop,mask=mask,start=np.array([x+w/2,y+h/2])))
    squares = sorted([o for o in objects if o['kind']=='square'],key=lambda o:o['w'])
    triangles = sorted([o for o in objects if o['kind']=='triangle'],key=lambda o:o['w'])
    # First collect each type in its own row, then sort those rows,
    # and finally bring both groups onto the same horizontal centerline.
    for o,x in zip([squares[0],squares[2],squares[1]],[220,470,780]):
        o['group']=np.array([x,350.])
    for o,x in zip([triangles[1],triangles[0],triangles[2]],[220,470,780]):
        o['group']=np.array([x,700.])
    for row,y in [(squares,350),(triangles,700)]:
        for o,x in zip(row,[220,470,780]):
            o['sorted']=np.array([x,float(y)])
            o['arc']=0.
    squares[2]['arc']=-125.
    squares[1]['arc']=125.
    triangles[1]['arc']=-95.
    triangles[0]['arc']=95.
    ordered=squares+triangles
    gaps=[24,24,48,24,24]
    total=sum(o['w'] for o in ordered)+sum(gaps)
    left=(1024-total)/2
    for i,o in enumerate(ordered):
        o['end']=np.array([left+o['w']/2,512.])
        left+=o['w']+(gaps[i] if i<5 else 0)
    (ROOT/'output').mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')],stdin=subprocess.PIPE)
    for frame in range(96):
        canvas=np.empty_like(original); canvas[:]=bg
        for o in objects:
            if frame<=32:
                t=smooth(frame/32)
                pos=o['start']*(1-t)+o['group']*t
            elif frame<=62:
                t=smooth((frame-32)/30)
                pos=o['group']*(1-t)+o['sorted']*t
                pos=pos+np.array([0,o['arc']*np.sin(np.pi*t)])
            else:
                t=smooth((frame-62)/33)
                pos=o['sorted']*(1-t)+o['end']*t
            x,y=np.rint(pos-np.array([o['w']/2,o['h']/2])).astype(int)
            region=canvas[y:y+o['h'],x:x+o['w']]
            region[o['mask']]=o['crop'][o['mask']]
        if frame==0:
            assert np.array_equal(canvas,original)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('ffmpeg failed')

if __name__=='__main__':
    main()
