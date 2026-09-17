import numpy as np, cv2, subprocess, os

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N_FRAMES = 16, 115

# Dot centres (x, y) in numerical order 1..8, measured from first_frame.png
DOTS = [(798, 459), (450, 508), (755, 666), (258, 291),
        (346, 866), (875, 229), (597, 256), (551, 843)]
COLOR = (0, 0, 255)  # red in BGR
THICK = 4

base = cv2.imread(SRC)
# full dot discs (radius ~45.5 px + anti-aliased rim): restored on top of the lines
keep = np.zeros(base.shape[:2], np.uint8)
for c in DOTS:
    cv2.circle(keep, c, 48, 1, -1)
keep = keep.astype(bool)
segs = list(zip(DOTS[:-1], DOTS[1:]))
n_seg = len(segs)
anim_frames = N_FRAMES - 1 - 2          # frame 0 untouched, 2 hold frames at end
per_seg = anim_frames / n_seg

def render(t):
    """t = frame index; returns frame with lines drawn up to that time."""
    img = base.copy()
    if t == 0:
        return img
    prog = min((t - 1 + 1) / per_seg, n_seg)   # segments completed (fractional)
    for i, (a, b) in enumerate(segs):
        f = np.clip(prog - i, 0, 1)
        if f <= 0:
            break
        end = (int(round(a[0] + (b[0] - a[0]) * f)), int(round(a[1] + (b[1] - a[1]) * f)))
        cv2.line(img, a, end, COLOR, THICK, cv2.LINE_AA)
    img[keep] = base[keep]
    return img

frames = [render(t) for t in range(N_FRAMES)]
assert np.array_equal(frames[0], base)

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                      '-s', '1024x1024', '-r', str(FPS), '-i', '-', '-sws_flags', 'accurate_rnd+full_chroma_int', '-c:v', 'libx264',
                      '-pix_fmt', 'yuv420p', '-crf', '12', '-r', str(FPS), OUT], stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
assert p.returncode == 0
print('wrote', OUT)
