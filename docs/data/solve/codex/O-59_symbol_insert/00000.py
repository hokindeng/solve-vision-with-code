from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
# Extract the original square and target outline without changing their rasterization.
square_mask=np.zeros(original.shape[:2],bool)
square_mask[465:560,623:718]=np.all(original[465:560,623:718]==[128,128,0],axis=2)
sy,sx=np.where(square_mask)
ref=original[19:137,888:1006]
mask=(ref[:,:,0]>220)&(ref[:,:,1]>70)&(ref[:,:,1]<210)&(ref[:,:,2]<80)
ry,rx=np.where(mask)
rx=rx+888
ry=ry+19
colors=original[ry,rx]
base=original.copy()
base[square_mask]=255

def smooth(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)

pipe=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','10','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(32):
    frame=base.copy()
    dx=round(105*smooth(i/12))
    frame[sy,sx+dx]=original[sy,sx]
    if i>=13:
        alpha=smooth((i-13)/7)
        cy=405+107*smooth((i-20)/10)
        xx=rx-946+670
        yy=ry-77+round(cy)
        frame[yy,xx]=np.round(frame[yy,xx]*(1-alpha)+colors*alpha).astype(np.uint8)
    if i==0:
        assert np.array_equal(frame,original)
    pipe.stdin.write(frame.tobytes())
pipe.stdin.close()
if pipe.wait()!=0:
    raise RuntimeError('ffmpeg failed')
