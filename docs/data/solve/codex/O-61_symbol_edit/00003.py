from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(original)
# Extract the original artwork exactly, including its edge pixels.
centers = [92 + 105*i for i in range(9)]
sprites = []
base = a.copy()
for cx in centers[:5]:
    patch = a[474:550, cx-38:cx+39].copy()
    mask = patch.max(axis=2) - patch.min(axis=2) > 20
    rgba = np.dstack([patch, mask.astype(np.uint8)*255])
    sprites.append(Image.fromarray(rgba, 'RGBA'))
    region = base[474:550, cx-38:cx+39]
    region[mask] = 255
base = Image.fromarray(base)
diamond = sprites[4]

def ease(v):
    v = max(0.0, min(1.0, v))
    return v*v*(3-2*v)

def put(canvas, sprite, x, scale=1):
    if scale <= 0:
        return
    if scale < 1:
        sprite = sprite.resize((max(1, round(77*scale)), max(1, round(76*scale))), Image.Resampling.LANCZOS)
    canvas.paste(sprite, (round(x-38+(77-sprite.width)/2), round(512-sprite.height/2)), sprite)

def frame(n):
    if n == 0:
        return a
    canvas = base.copy()
    first_shift = ease((n-5)/14)
    second_shift = ease((n-28)/13)
    for i, sprite in enumerate(sprites):
        put(canvas, sprite, centers[i] + 105*(first_shift+second_shift))
    put(canvas, diamond, centers[0], ease((n-19)/7))
    put(canvas, diamond, centers[1], ease((n-41)/8))
    result = np.array(canvas)
    # Keep the original grid strokes visible throughout the insertion.
    gray = (a.max(axis=2) == a.min(axis=2)) & (a[:,:,0] < 240)
    result[gray] = a[gray]
    return result

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
       '-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for n in range(54):
    proc.stdin.write(frame(n).tobytes())
proc.stdin.close()
errors = proc.stderr.read()
if proc.wait():
    raise RuntimeError(errors.decode())
