import cv2
import numpy as np
import os
import imageio

os.makedirs('/app/output', exist_ok=True)

img = cv2.imread('/app/first_frame.png')
# BGR to RGB for imageio
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# extract object and mask
obj_rect = img_rgb[96:165, 129:198].copy()
mask = np.any(obj_rect != [255, 255, 255], axis=-1)

# create clean background
bg = img_rgb.copy()
for x_start in [129, 373, 617, 861]:
    bg[96:165, x_start:x_start+69][mask] = [255, 255, 255]

a = 0.2816

def get_positions(obj_idx):
    surf_y = [472, 471, 483, 635][obj_idx]
    end_y = [794, 794, 794, 601][obj_idx]
    
    pos = []
    for f in range(80):
        y_air = 96 + a * (f**2)
        if y_air < surf_y:
            pos.append(y_air)
        else:
            break
            
    hit_frame = len(pos)
    frames_left = 80 - hit_frame
    
    if obj_idx < 3:
        # Sink
        sink_frames = 25
        for i in range(frames_left):
            if i < sink_frames:
                y = surf_y + (end_y - surf_y) * (i / sink_frames)
            else:
                y = end_y
            pos.append(y)
    else:
        # Float
        dip_frames = 10
        rise_frames = 15
        dip_y = surf_y + 20
        for i in range(frames_left):
            if i < dip_frames:
                t = i / dip_frames
                y = surf_y + (dip_y - surf_y) * (1 - (1-t)**2)
            elif i < dip_frames + rise_frames:
                t = (i - dip_frames) / rise_frames
                t2 = t**2 * (3 - 2*t)
                y = dip_y + (end_y - dip_y) * t2
            else:
                y = end_y
            pos.append(y)
            
    return [int(round(y)) for y in pos]

positions = [get_positions(i) for i in range(4)]
x_starts = [129, 373, 617, 861]

frames = []
for f in range(80):
    frame = bg.copy()
    for i in range(4):
        x = x_starts[i]
        y = positions[i][f]
        # ensure y is within bounds just in case (should be, max y=794, 794+69=863 < 1024)
        frame[y:y+69, x:x+69][mask] = obj_rect[mask]
    frames.append(frame)

# Save video
writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
for frame in frames:
    writer.append_data(frame)
writer.close()
