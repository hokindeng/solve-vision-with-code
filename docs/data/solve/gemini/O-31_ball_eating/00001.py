import cv2
import numpy as np
import math
import subprocess
import os

# Create output dir if not exists
os.makedirs('/app/output', exist_ok=True)

# Load the first frame
base_img = cv2.imread('/app/first_frame.png')

# Colors in BGR
COLOR_GREEN = [113, 179, 60]
COLOR_BLUE = [180, 130, 70]
COLOR_YELLOW = [0, 215, 255]
COLOR_ORANGE = [0, 140, 255]
COLOR_BLACK = [0, 0, 0]

# Positions
P0 = (642, 723)
P1 = (247, 164)
P2 = (863, 861)
P3 = (319, 759)
P4 = (759, 253)

# Radii
R0 = 23.5
R1 = 34.4
R2 = 46.5
R3 = 71.6
R4 = 98.7

def get_position(i):
    if i == 0:
        return P0
    elif i <= 24:
        t = i / 24.0
        return (P0[0] + t * (P1[0] - P0[0]), P0[1] + t * (P1[1] - P0[1]))
    elif i <= 57:
        t = (i - 24) / 33.0
        return (P1[0] + t * (P2[0] - P1[0]), P1[1] + t * (P2[1] - P1[1]))
    elif i <= 76:
        t = (i - 57) / 19.0
        return (P2[0] + t * (P3[0] - P2[0]), P2[1] + t * (P3[1] - P2[1]))
    elif i <= 100:
        t = (i - 76) / 24.0
        return (P3[0] + t * (P4[0] - P3[0]), P3[1] + t * (P4[1] - P3[1]))
    else:
        return P4

def get_radius(i):
    if i < 24:
        return R0
    elif i <= 28:
        t = (i - 24) / 4.0
        return R0 + t * (R1 - R0)
    elif i < 57:
        return R1
    elif i <= 61:
        t = (i - 57) / 4.0
        return R1 + t * (R2 - R1)
    elif i < 76:
        return R2
    elif i <= 80:
        t = (i - 76) / 4.0
        return R2 + t * (R3 - R2)
    elif i < 100:
        return R3
    elif i <= 104:
        t = (i - 100) / 4.0
        return R3 + t * (R4 - R3)
    else:
        return R4

# Prepare clean background (all original balls erased)
# Wait, actually we can just prepare an image with NO black ball, and erase others as we go
clean_base = base_img.copy()
clean_base[np.all(clean_base == COLOR_BLACK, axis=-1)] = [255, 255, 255]

out_frames = []

for i in range(108):
    if i == 0:
        # Exactly the first frame
        frame = base_img.copy()
    else:
        frame = clean_base.copy()
        
        # Erase eaten balls
        if i >= 24:
            frame[np.all(frame == COLOR_GREEN, axis=-1)] = [255, 255, 255]
        if i >= 57:
            frame[np.all(frame == COLOR_BLUE, axis=-1)] = [255, 255, 255]
        if i >= 76:
            frame[np.all(frame == COLOR_YELLOW, axis=-1)] = [255, 255, 255]
        if i >= 100:
            frame[np.all(frame == COLOR_ORANGE, axis=-1)] = [255, 255, 255]
            
        # Draw black ball
        pos = get_position(i)
        r = get_radius(i)
        x_sub = int(round(pos[0] * 16))
        y_sub = int(round(pos[1] * 16))
        r_sub = int(round(r * 16))
        cv2.circle(frame, (x_sub, y_sub), r_sub, (0, 0, 0), -1, lineType=cv2.LINE_8, shift=4)
        
    out_frames.append(frame)

# Save video using imageio/ffmpeg
import imageio
imageio.mimwrite('/app/output/video.mp4', [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in out_frames], fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

