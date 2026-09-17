from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
h, w = original.shape[:2]
yy, xx = np.indices((h,w))
blue = (original[:,:,2] == 255) & (original[:,:,0] == 0) & (original[:,:,1] == 0)
sample = blue & (xx < 400)
slope, intercept = np.polyfit(xx[sample], yy[sample], 1)
# Remove only the black arc and text, preserving the gray normal and blue ray.
black = np.all(original == 0, axis=2)
arc = black & (xx >= 418) & (xx <= 454) & (yy >= 377) & (yy <= 395)
text = (xx >= 500) & (xx <= 590) & (yy >= 375) & (yy <= 397) & np.any(original != 255, axis=2)
clean = original.copy()
clean[arc | text] = 255
# Restore the small part of the incident stroke hidden by the annotation.
clean[arc & (abs(yy - (slope*xx + intercept)) <= 1.1)] = (0,0,255)
origin = (453., float(slope*453 + intercept))
frames = 35
cmd = ['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(frames):
    if i == 0:
        frame = original.copy()
    else:
        fade = min(1., i/6.)
        frame = np.rint(original.astype(float)*(1-fade)+clean.astype(float)*fade).astype(np.uint8)
        progress = max(0., (i-3)/(frames-1-3))
        if progress:
            endx = origin[0] + (w-1-origin[0])*progress
            endy = origin[1] - slope*(endx-origin[0])
            mask = Image.new('L',(w,h),0)
            ImageDraw.Draw(mask).line([origin,(endx,endy)],fill=255,width=2)
            m = np.array(mask) > 0
            # Reflectivity scales the ray's contrast against the white background.
            reflected = np.array([8,8,255],dtype=np.uint8)
            frame[m] = reflected
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
Image.fromarray(frame).save(OUT/'last_frame.png')
