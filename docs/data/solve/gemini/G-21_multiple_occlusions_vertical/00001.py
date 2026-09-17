import cv2
import numpy as np
import imageio
import os

# Create output dir if not exists
os.makedirs('/app/output', exist_ok=True)

# Load first frame
first_frame = cv2.imread('/app/first_frame.png')
# Convert to RGB for imageio
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
height, width, _ = first_frame_rgb.shape

# Mask properties from first frame
mask_x, mask_y, mask_w, mask_h = 102, 30, 820, 251

# Extract the exact color of the mask
mask_color = tuple(int(c) for c in first_frame_rgb[mask_y, mask_x])

# Create base frame (background + objects, without the mask)
base_frame = first_frame_rgb.copy()
base_frame[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = (255, 255, 255)

# Video writer settings
num_frames = 58
fps = 16
out_path = '/app/output/video.mp4'

writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')

for i in range(num_frames):
    frame = base_frame.copy()
    
    # Calculate current mask y position
    current_y = int(round(mask_y + i * (height - mask_y) / (num_frames - 1)))
    
    # Draw mask
    cv2.rectangle(frame, (mask_x, current_y), (mask_x + mask_w - 1, current_y + mask_h - 1), mask_color, -1)
    
    writer.append_data(frame)

writer.close()
