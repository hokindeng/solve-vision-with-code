from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Extract only symbol pixels; the fixed cell borders and labels are untouched.
centers = [512, 617, 722, 827, 932]
layers = []
base = original.copy()
for x in centers:
    mask = np.zeros((1024, 1024), dtype=np.float32)
    crop = original[473:552, x-40:x+41]
    mask[473:552, x-40:x+41] = np.any(crop != 255, axis=2)
    rgb = original.astype(np.float32) * mask[:, :, None]
    layers.append((rgb, mask))
    base[mask > 0] = 255

def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame in range(45):
    if frame == 0:
        result = original
    else:
        result = base.astype(np.float32).copy()
        fade = 1 - smooth((frame-3)/16)
        shift = 105 * smooth((frame-20)/21)
        for index, (rgb, alpha) in enumerate(layers):
            if index == 0:
                a = alpha * fade
                color = rgb * fade
            else:
                transform = np.float32([[1, 0, -shift], [0, 1, 0]])
                a = cv2.warpAffine(alpha, transform, (1024, 1024), flags=cv2.INTER_LINEAR)
                color = cv2.warpAffine(rgb, transform, (1024, 1024), flags=cv2.INTER_LINEAR)
            result = result * (1-a[:, :, None]) + color
        result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    proc.stdin.write(result.tobytes())
proc.stdin.close()
errors = proc.stderr.read()
if proc.wait() != 0:
    raise RuntimeError(errors.decode())
