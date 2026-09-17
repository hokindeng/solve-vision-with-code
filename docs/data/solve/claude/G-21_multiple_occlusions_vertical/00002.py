import numpy as np, subprocess, os
from PIL import Image

W = H = 1024
FPS = 16
N = 58
first = np.array(Image.open('/app/first_frame.png').convert('RGB'))

MASK_COLOR = np.array([209, 209, 209], dtype=np.uint8)
m = np.all(first == MASK_COLOR, axis=2)
ys, xs = np.where(m)
x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
mh = y1 - y0

# background = scene without the mask (mask area is plain white background)
bg = first.copy()
bg[y0:y1, x0:x1] = 255

total = H - y0  # displacement needed for the mask top to leave the frame
os.makedirs('/app/output', exist_ok=True)
proc = subprocess.Popen(
    ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
     '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-r', str(FPS),
     '/app/output/video.mp4'],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for i in range(N):
    t = i / (N - 1)
    dy = int(round(t * total))
    frame = bg.copy() if i > 0 else first.copy()
    ty0, ty1 = y0 + dy, min(y1 + dy, H)
    if ty0 < H:
        frame[ty0:ty1, x0:x1] = MASK_COLOR
    proc.stdin.write(frame.tobytes())
proc.stdin.close(); proc.wait()
