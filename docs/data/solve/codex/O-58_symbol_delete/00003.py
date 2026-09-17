from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
h, w = source.shape[:2]
y, x = np.indices((h, w))
# Isolate only the colored symbols; the frames and position labels are fixed.
saturation = source.max(axis=2).astype(int) - source.min(axis=2).astype(int)
colored = saturation > 20
masks = [colored & (x >= a) & (x <= b) & (y >= 470) & (y <= 553)
         for a, b in [(524, 608), (645, 713), (731, 817)]]
base = source.copy()
for mask in masks:
    base[mask] = 255
layers = []
for mask in masks[1:]:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, :3] = source[mask]
    rgba[mask, 3] = 255
    layers.append(Image.fromarray(rgba))

def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '12', '-preset', 'slow',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame in range(45):
    if frame == 0:
        result = source
    else:
        fade = smooth((frame - 2) / 17)
        shift = 105 * smooth((frame - 20) / 22)
        canvas = base.copy()
        canvas[masks[0]] = np.rint(source[masks[0]] * (1 - fade) + 255 * fade).astype(np.uint8)
        image = Image.fromarray(canvas).convert('RGBA')
        for layer in layers:
            moved = layer.transform((w, h), Image.Transform.AFFINE,
                                    (1, 0, shift, 0, 1, 0),
                                    resample=Image.Resampling.BICUBIC)
            image = Image.alpha_composite(image, moved)
        result = np.array(image.convert('RGB'))
    proc.stdin.write(result.tobytes())
proc.stdin.close()
error = proc.stderr.read().decode()
if proc.wait() != 0:
    raise RuntimeError(error)
