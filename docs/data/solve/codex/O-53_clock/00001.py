from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
bg = np.array([240, 248, 255], dtype=np.float32)
h, w = source.shape[:2]
y, x = np.mgrid[:h, :w]
# Extract only the two colored hands; the face is retained from the input.
r, g, b = source.transpose(2, 0, 1).astype(float)
red = (r > g + 30) & (b > g) & (x > 500) & (x < 635) & (y > 245) & (y < 516)
brown = (r > g + 30) & (g > b + 20) & (x > 330) & (x < 515) & (y > 395) & (y < 516)
colors = [np.array([139,69,19],np.float32),np.array([220,20,60],np.float32)]
masks = []
base = source.copy()
for mask, color in zip([brown, red], colors):
    region = cv2.dilate(mask.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
    # The artwork has a solid background beneath the hands.
    delta = color-bg
    alpha = np.clip(np.sum((source.astype(np.float32)-bg)*delta,axis=2)/np.sum(delta*delta),0,1)
    alpha *= region
    alpha[(r==0)&(g==0)&(b==0)] = 0
    masks.append(alpha.astype(np.float32))
    base[region] = bg.astype(np.uint8)
# Retain the central pin exactly as supplied.
pin = (x-512)**2+(y-512)**2 <= 11**2
pin &= source.max(axis=2) < 30
base[pin] = source[pin]

cmd = ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
       '-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(120):
    if i == 0:
        frame = source
    else:
        t = i/119
        # Smooth acceleration and deceleration, with motion across the full clip.
        elapsed = 11*60*(t*t*(3-2*t))
        frame = base.astype(np.float32)
        for j, (alpha, color) in enumerate(zip(masks,colors)):
            degrees = elapsed*(0.5 if j==0 else 6)
            if i==119 and j==1:
                moved = alpha
            else:
                transform = cv2.getRotationMatrix2D((512,512),-degrees,1)
                moved = cv2.warpAffine(alpha,transform,(w,h),flags=cv2.INTER_LINEAR)
            frame = frame*(1-moved[:,:,None])+color*moved[:,:,None]
        frame = np.rint(frame).clip(0,255).astype(np.uint8)
        frame[pin] = source[pin]
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
