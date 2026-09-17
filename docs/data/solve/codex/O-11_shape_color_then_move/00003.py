from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Copy the original plus, including its outlines and overlapping bars.
plus = base[602:763, 115:276].copy()
blue = np.array([38, 63, 191])
green = np.array([95, 191, 111])
color_mask = np.all(plus == blue, axis=2)

def ease(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

def draw_plus(frame, x, y, recolor, opacity):
    patch = plus.copy()
    patch[color_mask] = np.rint(blue*(1-recolor)+green*recolor).astype(np.uint8)
    area = frame[y:y+161, x:x+161]
    area[:] = np.rint(area*(1-opacity)+patch*opacity).astype(np.uint8)

def erase_question(frame, cx, amount):
    area = frame[650:714, cx-23:cx+24]
    area[:] = np.rint(area*(1-amount)+255*amount).astype(np.uint8)

proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
    '-pixel_format','rgb24','-video_size','1024x1024','-framerate','16',
    '-i','-','-an','-c:v','libx264','-crf','0','-preset','medium',
    '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for i in range(60):
    frame = base.copy()
    # Introduce the middle plus, then demonstrate the color change.
    if i > 5:
        erase_question(frame, 482, ease((i-5)/5))
    if i >= 10:
        draw_plus(frame, 402, 602, ease((i-15)/13), ease((i-10)/5))
    # Introduce its green copy in the third slot and translate it down
    # by the same forty pixels as the star in the example.
    if i > 31:
        erase_question(frame, 769, ease((i-31)/5))
    if i >= 36:
        dy = round(40*ease((i-41)/13))
        draw_plus(frame, 689, 602+dy, 1, ease((i-36)/5))
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
