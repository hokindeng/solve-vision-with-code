from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The middle position is purple and the right position is pink.
# Recover the left unit's track under its movable handle.
track = base.copy()
track[648:705,176:233] = 0
track[672:681,200:209] = base[672:681,507:516]
light = np.all(base == (128,0,128), axis=2)
cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
       '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
       '-an','-c:v','libx264','-preset','slow','-crf','0',
       '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(24):
    if i == 0:
        frame = base.copy()
    else:
        t = np.clip((i-2)/19,0,1)
        eased = t*t*(3-2*t)
        x = round(176 + 82*eased)
        frame = track.copy()
        frame[648:705,x:x+57] = (128,128,128)
        if t >= 1:
            frame[light] = (255,192,203)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait():
    raise RuntimeError(err.decode())
