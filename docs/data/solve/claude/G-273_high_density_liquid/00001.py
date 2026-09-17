#!/usr/bin/env python3
"""Generate buoyancy video: 4 identical balls fall into 4 cups.
Cups 1-3 (light blue, less dense than ball): ball sinks to the bottom.
Cup 4 (dark green, denser than ball): ball floats at the surface.
Everything except the balls is copied unchanged from first_frame.png.
"""
import subprocess, numpy as np
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 80
BASE = np.array(Image.open('/app/first_frame.png').convert('RGB'))

# Geometry measured from first_frame.png
BALL_X = [163, 407, 651, 895]
BALL_Y0 = 130
BALL_BBOX = (129, 96, 198, 165)          # x0,y0,x1,y1 (exclusive) of ball 1
SURFACE_Y = [474, 473, 485, 637]         # top row of liquid in each cup
BOTTOM_Y = 863                            # last liquid row (inner bottom)
R = (BALL_BBOX[2] - BALL_BBOX[0]) / 2.0   # ~34.5
FLOATS = [False, False, False, True]

# Ball sprite (exact pixels + mask) extracted from the first frame
x0, y0, x1, y1 = BALL_BBOX
sprite = BASE[y0:y1, x0:x1].copy()
mask = (sprite != 255).any(-1)
# make mask a clean disc so antialiased edge pixels are included
yy, xx = np.mgrid[y0:y1, x0:x1]
mask |= ((xx + 0.5 - BALL_X[0]) ** 2 + (yy + 0.5 - BALL_Y0) ** 2) <= (R + 0.5) ** 2
SPR_H, SPR_W = mask.shape
SPR_OFF_Y = y0 - BALL_Y0
SPR_OFF_X = x0 - BALL_X[0]

# Background with the balls removed (their spots are white)
BG = BASE.copy()
for cx in BALL_X:
    sx, sy = cx + SPR_OFF_X, BALL_Y0 + SPR_OFF_Y
    reg = BG[sy:sy + SPR_H, sx:sx + SPR_W]
    reg[mask] = 255


def simulate(surface, floats):
    """Return list of ball centre y for each frame."""
    g = 1.08                    # px / frame^2 in air
    y, v = float(BALL_Y0), 0.0
    ys = []
    for f in range(N_FRAMES):
        ys.append(y)
        if y + R < surface:      # in the air
            v += g
            y += v
            if y + R > surface and not floats:
                pass
        elif not floats:
            # sinking: heavy drag, low terminal velocity
            v = v * 0.75 + 1.6
            y += v
            y = min(y, BOTTOM_Y - R)
            if y >= BOTTOM_Y - R:
                v = 0.0
        else:
            # floating: spring-damper toward equilibrium slightly below surface
            y_eq = surface + 9.0
            a = -0.12 * (y - y_eq) - 0.35 * v
            v += a
            y += v
    return ys


TRAJ = [simulate(SURFACE_Y[i], FLOATS[i]) for i in range(4)]


def render(f):
    img = BG.copy()
    for i, cx in enumerate(BALL_X):
        cy = int(round(TRAJ[i][f]))
        sx, sy = cx + SPR_OFF_X, cy + SPR_OFF_Y
        reg = img[sy:sy + SPR_H, sx:sx + SPR_W]
        reg[mask] = sprite[mask]
    return img


def main():
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'medium',
           '-r', str(FPS), '/app/output/video.mp4']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        frame = render(f)
        if f == 0:
            assert (frame == BASE).all(), 'first frame mismatch'
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(render(N_FRAMES - 1)).save('/app/output/last_frame.png')


if __name__ == '__main__':
    main()
