from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB')).copy()
# Preserve the exact contour supplied in the reference, including its inner arc.
patch = base[595:771, 100:281].copy()
mask_image = Image.new('L', (181,176), 0)
d = ImageDraw.Draw(mask_image)
d.ellipse((9,8,169,167), fill=255)
d.ellipse((59,32,169,143), fill=0)
fill_mask = np.asarray(mask_image).astype(float)/255
color = np.array([36, 197, 29.])

def ease(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3-2*t)

def erase_question(frame, cx, amount):
    sl = np.s_[651:711, cx-24:cx+25]
    frame[sl] = np.rint(base[sl]*(1-amount)+255*amount).astype(np.uint8)

def place(frame, dx, dy, opacity, fill):
    art = patch.astype(float).copy()
    # The filled area drains away while both original contours remain fixed.
    interior = fill_mask * fill
    white = np.all(patch > 245, axis=2)
    interior *= white
    art = art*(1-interior[:,:,None])+color*interior[:,:,None]
    y,x = 595+dy,100+dx
    dest = frame[y:y+176,x:x+181].astype(float)
    frame[y:y+176,x:x+181] = np.rint(dest*(1-opacity)+art*opacity).astype(np.uint8)

command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-crf','15','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',
           str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for i in range(64):
    t=i/16
    frame=base.copy()
    if i:
        reveal=ease((t-.20)/.45)
        erase_question(frame,476,reveal)
        place(frame,288,0,reveal,1-ease((t-.75)/1.0))
        reveal_right=ease((t-2.0)/.4)
        erase_question(frame,764,reveal_right)
        shift=round(60*ease((t-2.45)/1.25))
        place(frame,576,shift,reveal_right,0)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
