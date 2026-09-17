import cv2
import numpy as np
import subprocess
import os

os.makedirs('/app/output', exist_ok=True)

img = cv2.imread('/app/first_frame.png')

# b2 region
b2_patch = img[440:580, 270:400].copy()

# original ? region
q_patch = img[440:580, 793:923].copy()

num_frames = 25

cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', '1024x1024',
    '-pix_fmt', 'bgr24',
    '-r', '16',
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '/app/output/video.mp4'
]

process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

for i in range(num_frames):
    alpha = i / (num_frames - 1)
    blended_patch = cv2.addWeighted(q_patch, 1 - alpha, b2_patch, alpha, 0)
    frame = img.copy()
    frame[440:580, 793:923] = blended_patch
    process.stdin.write(frame.tobytes())

process.stdin.close()
process.wait()
