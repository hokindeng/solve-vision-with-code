#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: three identical objects fall into three cups.
Cups 1 and 2 (light liquid, lower density than the object) -> the object sinks to the bottom.
Cup 3 (dark liquid, higher density than the object) -> the object floats.
Everything except the three objects stays pixel-identical to first_frame.png."""
import subprocess, numpy as np
from PIL import Image

W = H = 1024
FPS, N_FRAMES = 16, 80
SUB = 20                    # physics sub-steps per frame
BASE = np.array(Image.open('/app/first_frame.png').convert('RGB'))

SIZE = 57                   # object side (px)
FILL, BORDER = (243, 156, 18), (211, 84, 0)
Y0 = 94                     # object top in first frame
OBJ_X = [240, 484, 728]     # object left edges
SURFACE = [495, 455, 590]   # first liquid row per cup
CUP_BOTTOM = 843            # last interior row of each cup
FLOATS = [False, False, True]

G = 1.12                    # px / frame^2 in air
V_TERM = 9.0                # terminal sinking speed in liquid (px / frame)
K_SINK = 0.5                # velocity relaxation rate in liquid (1 / frame)
SUBMERGED_EQ = 0.6          # object density / dark liquid density (fraction submerged when floating)
K_FLOAT = 0.3              # linear drag for the floating object

def simulate(i):
    """Return list of object-top y (float) for each frame."""
    y, v = float(Y0), 0.0
    dt = 1.0 / SUB
    ys = []
    for f in range(N_FRAMES):
        ys.append(y)
        for _ in range(SUB):
            bottom = y + SIZE
            depth = np.clip(bottom - SURFACE[i], 0, SIZE)      # submerged height
            if depth <= 0:
                a = G
            elif FLOATS[i]:
                a = G * (1.0 - (depth / SIZE) / SUBMERGED_EQ) - K_FLOAT * v
            else:
                a = -K_SINK * (v - V_TERM)
            v += a * dt
            y += v * dt
            if not FLOATS[i] and y + SIZE >= CUP_BOTTOM:
                y, v = CUP_BOTTOM - SIZE, 0.0
    return ys

def final_top(i):
    if FLOATS[i]:
        return SURFACE[i] + SUBMERGED_EQ * SIZE - SIZE
    return CUP_BOTTOM - SIZE

def draw_object(img, x, y):
    y = int(round(y))
    img[y:y + SIZE, x:x + SIZE] = BORDER
    img[y + 1:y + SIZE - 1, x + 1:x + SIZE - 1] = FILL

def main():
    background = BASE.copy()
    for x in OBJ_X:                     # erase objects from the static background
        background[Y0:Y0 + SIZE, x:x + SIZE] = (255, 255, 255)
    traj = [simulate(i) for i in range(3)]
    # blend the last few frames to the exact equilibrium so the final frame is settled
    for i in range(3):
        for f in range(N_FRAMES - 8, N_FRAMES):
            t = (f - (N_FRAMES - 9)) / 8.0
            traj[i][f] = (1 - t) * traj[i][f] + t * final_top(i)
        traj[i][-1] = final_top(i)

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow',
           '-r', str(FPS), '/app/output/video.mp4']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        frame = BASE.copy() if f == 0 else background.copy()
        if f > 0:
            for i in range(3):
                draw_object(frame, OBJ_X[i], traj[i][f])
        p.stdin.write(frame.tobytes())
    p.stdin.close(); p.wait()
    assert p.returncode == 0
    for i in range(3):
        print(f'object {i}: ' + ' '.join(f'{v:.0f}' for v in traj[i][::8]) + f' -> {traj[i][-1]:.1f}')

if __name__ == '__main__':
    main()
