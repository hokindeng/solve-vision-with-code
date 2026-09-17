from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Rectangles are obtained directly from connected nonwhite components,
    # bounded above the shelf so that the shelf is never part of a sprite.
    import cv2
    mask = np.all(original[:513] == [175,175,21], axis=2).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = sorted(stats[1:], key=lambda r: r[0])
    moves = [(int(x)-2,int(y)-2,int(w)+4,int(h)+4,dest)
             for (x,y,w,h,area),dest in zip(boxes,[214,109,423])]
    frames = [original.copy()]
    settled = original.copy()
    for x,y,w,h,dest in moves:
        sprite = original[y:y+h,x:x+w].copy()
        base = settled.copy()
        base[y:y+h,x:x+w] = 255
        base[512:y+h,x:x+w] = original[512,0]
        # Lift clear of all standing books before translating across the shelf.
        high = 50
        poses = [(x,round((y+high)/2)), (x,high),
                 (round(x+(dest-x)/3),high),
                 (round(x+2*(dest-x)/3),high), (dest,high),
                 (dest,round((y+high)/2)), (dest,y)]
        for px,py in poses:
            frame=base.copy()
            frame[py:py+h,px:px+w]=sprite
            frames.append(frame)
        settled = frames[-1].copy()
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo',
        '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
        '-an','-c:v','libx264','-crf','0','-preset','slow',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__=='__main__':
    main()
