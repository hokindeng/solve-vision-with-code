from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Each row shifts the three symbols one position left: square, plus, pentagon.
# Copy the existing pentagon pixels, preserving its exact stroke and color.
source = base[420:590, 85:255]
mask = (source[:,:,0] > 200) & (source[:,:,1] > 60) & (source[:,:,1] < 200) & (source[:,:,2] < 60)
sy, sx = np.where(mask)
ty, tx = sy + 761, sx + 767
colors = source[sy, sx]
angles = np.mod(np.arctan2(tx - 852, -(ty - 852)), 2*np.pi)
# The only original pixels removed are those belonging to the question mark.
question = np.zeros(base.shape[:2], dtype=bool)
question[790:915, 810:895] = np.any(base[790:915,810:895] != 255, axis=2)
command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
p = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(35):
    frame = base.copy()
    fade = min(i / 8, 1)
    frame[question] = np.rint(base[question].astype(float)*(1-fade) + 255*fade).astype(np.uint8)
    progress = np.clip((i-8)/25, 0, 1)
    if progress > 0:
        visible = angles <= progress*2*np.pi
        frame[ty[visible],tx[visible]] = colors[visible]
    assert np.array_equal(frame[:683], base[:683])
    assert np.array_equal(frame[683:,:683], base[683:,:683])
    p.stdin.write(frame.tobytes())
p.stdin.close()
err = p.stderr.read()
if p.wait():
    raise RuntimeError(err.decode())
