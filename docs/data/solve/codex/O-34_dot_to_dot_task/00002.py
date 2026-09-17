from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Numbered dots, in numerical order. The original dots stay above the lines.
points = [(889,155), (424,575), (171,391), (783,649), (654,819), (631,227), (316,186)]
protected = np.any(base < 250, axis=2)
# Include the antialiased circle edges in the protected region.
protected = cv2.dilate(protected.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame_index in range(100):
    mask = np.zeros(base.shape[:2], dtype=np.uint8)
    progress = max(0.0, min(6.0, (frame_index - 4) / 15.0))
    for i in range(6):
        fraction = min(1.0, max(0.0, progress - i))
        if fraction <= 0:
            continue
        start = points[i]
        end = tuple(round(start[j] + fraction * (points[i+1][j] - start[j])) for j in range(2))
        cv2.line(mask, start, end, 255, 5, cv2.LINE_AA)
    mask[protected] = 0
    alpha = mask[...,None].astype(np.float32) / 255
    frame = np.rint(base * (1-alpha) + np.array([255,0,0]) * alpha).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
error = proc.stderr.read()
if proc.wait() != 0:
    raise RuntimeError(error.decode())
print(OUT/'video.mp4')
