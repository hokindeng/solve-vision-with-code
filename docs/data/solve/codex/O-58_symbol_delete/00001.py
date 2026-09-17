from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
h, w = source.shape[:2]
# Extract only the symbols, leaving the numbered boxes untouched.
target = np.zeros((h, w), dtype=bool)
target[474:552, 368:447] = np.any(source[474:552, 368:447] != 255, axis=2)
moving = np.zeros((h, w), dtype=bool)
for cx in (512, 617, 722, 827, 932):
    moving[474:552, cx-39:cx+40] = np.any(source[474:552, cx-39:cx+40] != 255, axis=2)
base = source.copy()
base[target | moving] = 255
# Premultiplied ink allows smooth subpixel translations on white.
ink = np.zeros_like(source, dtype=np.float32)
ink[moving] = 255.0-source[moving].astype(np.float32)
target_ink = np.zeros_like(ink)
target_ink[target] = 255.0-source[target].astype(np.float32)

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t*t*(3-2*t)

command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(45):
    fade = smooth(i/18)
    shift = 105.0*smooth((i-18)/24)
    translated = cv2.warpAffine(ink, np.float32([[1, 0, -shift], [0, 1, 0]]),
                                (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame = np.clip(base.astype(np.float32)-translated-target_ink*(1-fade), 0, 255)
    frame = np.rint(frame).astype(np.uint8)
    if i == 0:
        assert np.array_equal(frame, source)
    process.stdin.write(frame.tobytes())
process.stdin.close()
errors = process.stderr.read()
if process.wait() != 0:
    raise RuntimeError(errors.decode())
print(OUT/'video.mp4')
