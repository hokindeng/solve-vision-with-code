from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The middle unit demonstrates that the left lever position means red.
# Reconstruct the black surface and concealed middle detent only under
# each moving lever. All other pixels are copied from the source image.
dot = original[672:681, 507:516].copy()

def frame_at(index):
    frame = original.copy()
    for x, center, start, end in [(176,204,2,11), (790,818,12,22)]:
        t = min(1.0, max(0.0, (index-start)/(end-start)))
        if t <= 0:
            continue
        eased = t*t*(3-2*t)
        frame[648:705,x:x+57] = 0
        frame[672:681,center-4:center+5] = dot
        moving_x = x - round(82*eased)
        frame[648:705,moving_x:moving_x+57] = 128
        if t == 1:
            yy,xx = np.indices(original.shape[:2])
            mask = (original[:,:,0]==255)&(original[:,:,1]==0)&(original[:,:,2]==255)&(abs(xx-center)<60)
            frame[mask] = [255,0,0]
    return frame

proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(24):
    proc.stdin.write(frame_at(i).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Video encoding failed')
