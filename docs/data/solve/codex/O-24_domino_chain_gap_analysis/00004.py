from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Preserve the original artwork, including the numerals, on each moving tile.
boxes = [(145, 588, 191, 722), (204, 588, 250, 722), (265, 588, 311, 722)]
background = source.copy()
sprites = []
for x0,y0,x1,y1 in boxes:
    sprites.append(source[y0:y1,x0:x1].copy())
    background[y0:720,x0:x1] = 255
    for y in range(720,y1):
        background[y,x0:x1] = source[y,x0-1]

# Successive contacts start each motion. Domino 3 cannot bridge the gap.
starts = [3, 12, 22]
ends = [38, 41, 45]
angles = [60.0, math.degrees(math.acos(44/133)), 90.0]

def frame_at(i):
    if i == 0:
        return source.copy()
    frame = background.copy()
    # Rear tiles are composited first so the earlier tiles rest on top.
    for j in [2,1,0]:
        x0,y0,x1,y1 = boxes[j]
        t = np.clip((i-starts[j])/(ends[j]-starts[j]),0,1)
        theta = math.radians(angles[j] * (t*t*(3-2*t)))
        if theta == 0:
            frame[y0:y1,x0:x1] = sprites[j]
            continue
        c,s = math.cos(theta),math.sin(theta)
        w,h = x1-x0,y1-y0
        px,py = x1-1,y1-1
        transform = np.float32([[c,-s,px-c*(w-1)+s*(h-1)],
                                [s,c,py-s*(w-1)-c*(h-1)]])
        # Supersampling softens just the moving silhouette.
        scale = 3
        big_transform = transform.copy()
        big_transform[:,2] *= scale
        tile = cv2.resize(sprites[j],None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        alpha = np.full(tile.shape[:2],255,np.uint8)
        warped = cv2.warpAffine(tile,big_transform,(1024*scale,1024*scale),flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(alpha,big_transform,(1024*scale,1024*scale),flags=cv2.INTER_LINEAR)
        warped = cv2.resize(warped,(1024,1024),interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(1024,1024),interpolation=cv2.INTER_AREA)
        a = mask[:,:,None].astype(np.float32)/255
        # The warped RGB is already premultiplied by its edge coverage.
        frame = np.clip(warped.astype(np.float32)+frame*(1-a),0,255).astype(np.uint8)
    return frame

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18',
       '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
p = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(50):
    frame = frame_at(i)
    p.stdin.write(frame.tobytes())
p.stdin.close()
if p.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
