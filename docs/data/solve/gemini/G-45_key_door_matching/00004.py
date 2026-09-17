import cv2
import numpy as np
import imageio

img = cv2.imread('/app/first_frame.png')

# 1. Get Agent Shape and clean background
green_mask = cv2.inRange(img, np.array([0, 255, 0]), np.array([0, 255, 0]))
agent_coords = np.argwhere(green_mask > 0)
agent_y, agent_x = agent_coords[:, 0], agent_coords[:, 1]
M = cv2.moments(green_mask)
agent_cx = int(M['m10']/M['m00'])
agent_cy = int(M['m01']/M['m00'])
rel_y = agent_y - agent_cy
rel_x = agent_x - agent_cx

bg_with_key = img.copy()
bg_with_key[green_mask > 0] = [255, 255, 255] # Remove agent from background

bg_without_key = bg_with_key.copy()
key_cx = int(93 + 4 * 93 + 46.5)
key_cy = int(93 + 2 * 93 + 46.5)
key_cell_x = 93 + 4 * 93
key_cell_y = 93 + 2 * 93
key_cell = bg_without_key[key_cell_y:key_cell_y+93, key_cell_x:key_cell_x+93]
yellow_mask = cv2.inRange(key_cell, np.array([0, 255, 255]), np.array([0, 255, 255]))
key_cell[yellow_mask > 0] = [255, 255, 255]

# 2. Define Path
path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4), (2, 5), (2, 6), (3, 6), (4, 6), (4, 7), (4, 8), (5, 8), (6, 8), (7, 8)]

def get_center(r, c):
    cx = int(93 + c * 93 + 46.5)
    cy = int(93 + r * 93 + 46.5)
    return cx, cy

# 3. Generate Frames
num_frames = 52
total_segments = len(path) - 1

frames = []
for f in range(num_frames):
    # d goes from 0 to total_segments
    d = f * total_segments / (num_frames - 1)
    i = int(d)
    if i >= total_segments:
        i = total_segments - 1
        t = 1.0
    else:
        t = d - i
        
    r1, c1 = path[i]
    r2, c2 = path[i+1]
    
    cx1, cy1 = get_center(r1, c1)
    cx2, cy2 = get_center(r2, c2)
    
    new_cx = int(round(cx1 * (1 - t) + cx2 * t))
    new_cy = int(round(cy1 * (1 - t) + cy2 * t))
    
    if d >= 10.0:
        frame = bg_without_key.copy()
    else:
        frame = bg_with_key.copy()
        
    # Draw Agent
    frame[new_cy + rel_y, new_cx + rel_x] = [0, 255, 0]
    
    # Convert BGR to RGB for imageio
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

# 4. Save Video
import os
os.makedirs('/app/output', exist_ok=True)
writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
for frame in frames:
    writer.append_data(frame)
writer.close()
