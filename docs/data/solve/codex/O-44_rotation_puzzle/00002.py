from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
blue = np.array([59,130,246], dtype=np.uint8)
mask = np.all(source == blue, axis=2)
background = source.copy()
background[mask] = 255
# Centers are fixed. The lower elbows make one complete turn so that all
# four pipes move simultaneously and finish at the same instant.
centers = [(387,387), (637,387), (387,637), (637,637)]
turns = [-168,78,360,-360]
pipe_masks = []
for cx,cy in centers:
    pipe_masks.append((mask[cy-108:cy+109,cx-108:cx+109]*255).astype(np.uint8))
command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-preset','slow','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',
           str(OUT/'video.mp4')]
process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame_index in range(96):
    if frame_index == 0:
        frame = source.copy()
    else:
        t = frame_index/95
        eased = t*t*(3-2*t)
        frame = background.copy()
        for (cx,cy), pipe, turn in zip(centers,pipe_masks,turns):
            # OpenCV uses counterclockwise positive angles.
            transform = cv2.getRotationMatrix2D((108,108), -turn*eased, 1)
            alpha = cv2.warpAffine(pipe,transform,(217,217),flags=cv2.INTER_LINEAR)
            roi = frame[cy-108:cy+109,cx-108:cx+109]
            a = alpha[:,:,None].astype(np.float32)/255
            roi[:] = np.rint(roi*(1-a)+blue*a).astype(np.uint8)
    process.stdin.write(frame.tobytes())
process.stdin.close()
error = process.stderr.read()
if process.wait():
    raise RuntimeError(error.decode())
Image.fromarray(frame).save(OUT/'last_frame.png')
