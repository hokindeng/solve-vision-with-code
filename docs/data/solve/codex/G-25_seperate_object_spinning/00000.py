from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = cv2.imread(str(ROOT / 'first_frame.png'))
h, w = original.shape[:2]
# Isolate each complete object, including its original fine boundary pixels.
mask = (original.min(axis=2) < 254).astype(np.uint8)
mask[:, 500:] = 0
count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
background = original.copy()
background[labels > 0] = 255
# Keep all dashed outline pixels fixed, even where an arriving edge meets them.
outline = (original.min(axis=2) < 255)
outline[:, :500] = False
# Image-coordinate angles increase clockwise. All destination centers retain y.
objects = [
    (1, (172., 433.), 676., 0.),
    (2, (391.5, 432.5), 850.5, 29.),
    (3, (371., 625.5), 676., 8.),
    (4, (170.5, 625.5), 803.5, 12.),
]
layers = []
for label, center, destination_x, rotation in objects:
    delta = np.zeros_like(original)
    select = labels == label
    delta[select] = 255 - original[select]
    layers.append((delta, center, destination_x, rotation))

def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3.-2.*t)

proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', '16', '-i', '-',
    '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'slow',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for i in range(48):
    if i == 0:
        frame = original.copy()
    else:
        # Rotate together in place, then translate together to arrive on frame 47.
        rotation_progress = smooth(i / 17.)
        movement_progress = smooth((i - 17.) / 30.)
        frame = background.astype(np.int16)
        for delta, center, destination_x, rotation in layers:
            matrix = cv2.getRotationMatrix2D(center, -rotation * rotation_progress, 1.)
            matrix[0, 2] += (destination_x - center[0]) * movement_progress
            moved = cv2.warpAffine(delta, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            frame -= moved.astype(np.int16)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame[outline] = original[outline]
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
