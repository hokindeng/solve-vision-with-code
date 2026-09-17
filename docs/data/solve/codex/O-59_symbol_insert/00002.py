from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = Image.open(ROOT / 'first_frame.png').convert('RGB')

def sprite(box):
    a = np.array(original.crop(box))
    alpha = np.where(np.any(a != 255, axis=2), 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack((a, alpha)))

pink_box = (639, 482, 699, 542)
gray_box = (734, 472, 814, 552)
pink = sprite(pink_box)
gray = sprite(gray_box)
yellow = sprite((907, 38, 985, 116))
base = original.copy()
base.paste('white', pink_box)
base.paste('white', gray_box)

def ease(t):
    t = min(1., max(0., t))
    return t*t*(3-2*t)

def frame(i):
    if i == 0:
        return original.copy()
    canvas = base.copy()
    # Shift the two occupied slots into slots seven and eight.
    shift = round(105 * ease(i / 13))
    canvas.paste(pink, (639 + shift, 482), pink)
    canvas.paste(gray, (734 + shift, 472), gray)
    # Introduce the reference symbol above the newly emptied sixth slot.
    if i >= 14:
        opacity = ease((i - 14) / 6)
        descent = ease((i - 20) / 11)
        obj = yellow.copy()
        obj.putalpha(obj.getchannel('A').point(lambda a: round(a * opacity)))
        canvas.paste(obj, (630, round(352 + 121 * descent)), obj)
    return canvas

proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
    '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for i in range(32):
    proc.stdin.write(np.asarray(frame(i)).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Video encoding failed')
