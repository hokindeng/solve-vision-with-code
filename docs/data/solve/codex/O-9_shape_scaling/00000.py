from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
yellow = ((a[:,:,0] > 200) & (a[:,:,1] > 200) & (a[:,:,2] < 100)).astype(np.uint8)
_, labels, stats, centers = cv2.connectedComponentsWithStats(yellow)
# Comparing filled areas avoids one-pixel bounding-box rounding.
scale = float(np.sqrt(stats[2, cv2.CC_STAT_AREA] / stats[1, cv2.CC_STAT_AREA]))
# The only mutable region contains the missing answer.
box = (700, 690, 839, 841)
original = base.crop(box)
blank = Image.new('RGB', original.size, 'white')
verts = np.array([(256,694),(326,745),(299,829),(211,829),(184,745)], dtype=float)
relative = verts - np.array([256.,768.])

def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3-2*t)

proc = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
    '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
    '-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
    '-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for i in range(60):
    frame = base.copy()
    if i:
        # Erase the question mark gently, then grow the scaled answer.
        patch = Image.blend(original, blank, float(smooth(i / 15)))
        if i >= 15:
            progress = smooth((i-15)/40)
            factor = scale * progress
            if factor > 0:
                hi = Image.new('RGB',(patch.width*4,patch.height*4),'white')
                d = ImageDraw.Draw(hi)
                points = (relative * factor + np.array([769-box[0],768-box[1]]))*4
                d.polygon([tuple(p) for p in points], fill=(255,255,0))
                d.line([tuple(p) for p in np.vstack([points,points[0]])], fill='black',width=3)
                patch = hi.resize(patch.size,Image.Resampling.LANCZOS)
        frame.paste(patch,box[:2])
    proc.stdin.write(np.asarray(frame).tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
print(f'Wrote {OUT / "video.mp4"}; measured scale = {scale:.5f}')
