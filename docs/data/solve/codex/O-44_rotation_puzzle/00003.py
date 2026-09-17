from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
pipe_color = np.array([132, 204, 22], dtype=np.float32)
pipe_pixels = np.all(original == pipe_color.astype(np.uint8), axis=2)
background = original.copy()
background[pipe_pixels] = 255
# Rotation pivots remain at the centers of their original tiles.
# Angles are clockwise in image coordinates. All final elbows face inward.
tiles = [(387,387,0,-360), (637,387,-78,90),
         (387,637,-9,-90), (637,637,27,180)]
scale = 4
radius = 109
size = radius * 2
layers = []
for cx, cy, start, end in tiles:
    x, y = cx-radius, cy-radius
    mask = pipe_pixels[y:y+size, x:x+size].astype(np.float32)
    large = cv2.resize(mask, (size*scale,size*scale), interpolation=cv2.INTER_NEAREST)
    layers.append((x,y,start,end,large))

command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
           '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
           '-an','-c:v','libx264','-crf','16','-preset','medium',
           '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
process = subprocess.Popen(command, stdin=subprocess.PIPE)
try:
    for frame_index in range(96):
        if frame_index == 0:
            frame = original.copy()
        else:
            t = frame_index / 95
            # Quintic easing spreads the action across the entire six seconds.
            progress = t*t*t*(10 + t*(-15 + 6*t))
            frame = background.copy()
            for x,y,start,end,large in layers:
                angle = (end-start)*progress
                pivot = (radius*scale+(scale-1)/2, radius*scale+(scale-1)/2)
                matrix = cv2.getRotationMatrix2D(pivot, -angle, 1)
                rotated = cv2.warpAffine(large, matrix, (size*scale,size*scale),
                                         flags=cv2.INTER_LINEAR, borderValue=0)
                alpha = cv2.resize(rotated, (size,size), interpolation=cv2.INTER_AREA)[...,None]
                patch = frame[y:y+size,x:x+size]
                patch[:] = np.rint(patch*(1-alpha)+pipe_color*alpha).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoder failed')
except BaseException:
    process.kill()
    raise
