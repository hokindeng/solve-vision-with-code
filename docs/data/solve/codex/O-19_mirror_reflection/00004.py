from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
h, w = base.shape[:2]
y, x = np.indices((h, w))
# The arc and label are the only original pixels to erase. Keep the normal.
gray = (base[:,:,0] == base[:,:,1]) & (base[:,:,1] == base[:,:,2])
arc = (x >= 524) & (x <= 564) & (y >= 569) & (y <= 598) & gray & (base[:,:,0] < 100)
label = (x >= 605) & (x <= 703) & (y >= 563) & (y <= 594)
annotation = arc | label
origin = np.array([562., 612.])
# Reflection reverses the vertical component of the incoming direction.
end = np.array([1023., 612. - (1023.-562.) * (612.-378.)/562.])
direction = end-origin
unit = direction / np.linalg.norm(direction)
normal = np.array([-unit[1],unit[0]])
scale=4
frames=[]
for i in range(35):
    frame=base.copy()
    if i:
        fade=min(1.,i/6.)
        frame[annotation]=np.rint(base[annotation]*(1-fade)+255*fade).astype(np.uint8)
        progress=max(0.,min(1.,(i-3)/31.))
        if progress:
            mask=Image.new('L',(w*scale,h*scale),0)
            draw=ImageDraw.Draw(mask)
            point=origin+direction*progress
            def coords(p): return tuple(np.rint(p*scale).astype(int))
            draw.line([coords(origin),coords(point)],fill=255,width=scale+1)
            # A direction arrow emerges when the advancing ray reaches it.
            arrow_t=.65
            if progress >= arrow_t:
                tip=origin+direction*arrow_t
                size=15*min(1.,(progress-arrow_t)/.05)
                draw.line([coords(tip-unit*size+normal*size*.40),coords(tip),coords(tip-unit*size-normal*size*.40)],fill=255,width=scale+1)
            alpha=np.array(mask.resize((w,h),Image.Resampling.LANCZOS),dtype=float)/255*.38
            # Blend the reflected blue light against the unchanged scene.
            frame=np.rint(frame*(1-alpha[:,:,None])+np.array([0,0,255])*alpha[:,:,None]).astype(np.uint8)
    frames.append(frame)
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','12','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for frame in frames:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
errors=proc.stderr.read()
if proc.wait(): raise RuntimeError(errors.decode())
Image.fromarray(frames[-1]).save(OUT/'last_frame.png')
