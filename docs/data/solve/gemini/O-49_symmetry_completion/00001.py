import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    # smoothstep
    return t * t * (3 - 2 * t)

img = cv2.imread('/app/first_frame.png')
# Convert BGR to RGB for imageio
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, _ = img.shape

missing_cells = [(1, 4), (1, 5), (4, 5), (5, 4)]

xs = [(205, 305), (308, 407), (410, 510), (513, 612), (615, 714), (717, 817)]
ys = [(205, 305), (308, 407), (410, 510), (513, 612), (615, 714), (717, 817)]

out_path = '/app/output/video.mp4'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

writer = imageio.get_writer(out_path, fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')

total_frames = 35
anim_frames = 30

for frame_idx in range(total_frames):
    frame = img.copy()
    
    if frame_idx == 0:
        p = 0.0
    elif frame_idx >= anim_frames:
        p = 1.0
    else:
        p = frame_idx / float(anim_frames)
        
    p = ease_in_out(p)
        
    for r, c in missing_cells:
        x1, x2 = xs[c]
        y1, y2 = ys[r]
        
        cell_w = x2 - x1
        cell_h = y2 - y1
        
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        curr_w = cell_w * p
        curr_h = cell_h * p
        
        curr_x1 = int(round(cx - curr_w / 2))
        curr_x2 = int(round(cx + curr_w / 2))
        curr_y1 = int(round(cy - curr_h / 2))
        curr_y2 = int(round(cy + curr_h / 2))
        
        curr_x1 = max(x1, curr_x1)
        curr_x2 = min(x2, curr_x2)
        curr_y1 = max(y1, curr_y1)
        curr_y2 = min(y2, curr_y2)
        
        if curr_x2 > curr_x1 and curr_y2 > curr_y1:
            region = frame[curr_y1:curr_y2, curr_x1:curr_x2]
            # Replace white [255, 255, 255] with blue [37, 99, 235] in RGB
            mask = (region[:, :, 0] == 255) & (region[:, :, 1] == 255) & (region[:, :, 2] == 255)
            region[mask] = [37, 99, 235]
            
    writer.append_data(frame)

writer.close()
print("Done")

