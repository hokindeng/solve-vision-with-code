from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import cv2

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The repeating cycle is small, large, medium. Copy the medium diamond
# exactly, placing its center at the center of the answer box.
patch = base[480:545, 400:476].copy()
source_mask = np.any(patch != 255, axis=2)
mask = np.zeros((1024,1024), np.uint8)
mask[480:545,845:921] = source_mask
result = base.copy()
region = result[480:545,845:921]
region[source_mask] = patch[source_mask]
interior = cv2.erode(mask, np.ones((3,3),np.uint8))
outline = (mask > 0) & (interior == 0)
y,x = np.indices(mask.shape)
# Start at the top vertex and trace clockwise around all four edges.
angle = np.mod(np.arctan2(x-882.5, -(y-512.0)), 2*np.pi)
encoder = subprocess.Popen([
    'ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
    '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
    '-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
    '-movflags','+faststart',str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame_id in range(60):
    frame = base.copy()
    if 8 <= frame_id < 32:
        progress = (frame_id-7)/24
        visible = outline & (angle <= progress*2*np.pi)
        frame[visible] = result[visible]
    elif frame_id >= 32:
        progress = min(1.0, (frame_id-31)/20)
        visible = outline | ((mask > 0) & (y <= 483 + progress*58))
        frame[visible] = result[visible]
    encoder.stdin.write(frame.tobytes())
encoder.stdin.close()
if encoder.wait() != 0:
    raise RuntimeError('Video encoding failed')
