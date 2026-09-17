import cv2
import numpy as np
import imageio
import os

# 1. Read first frame and extract background & shapes
img = cv2.imread('/app/first_frame.png')
# convert to RGB for imageio
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

bg = img_rgb.copy()
bg[np.all(bg == [0, 255, 0], axis=-1)] = [255, 255, 255]
bg[np.all(bg == [0, 255, 255], axis=-1)] = [255, 255, 255]
bg[np.all(bg == [255, 128, 0], axis=-1)] = [255, 255, 255]

def get_shape(color):
    mask = np.all(img_rgb == color, axis=-1)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2
    rel_coords = coords - np.array([center_y, center_x])
    return rel_coords, (center_y, center_x)

agent_shape, agent_center = get_shape([0, 255, 0])
k1_shape, k1_center = get_shape([0, 255, 255])
k2_shape, k2_center = get_shape([255, 128, 0])

# Grid to pixel mapping
def grid_to_pixel(r, c):
    y = int(93.0909 * r + 93.0909 / 2)
    x = int(93.0909 * c + 93.0909 / 2)
    return y, x

# 2. Define path
full_path = [
    (5, 8), (5, 7), (4, 7), (3, 7), (3, 6), (3, 5), (4, 5), # Agent to K2 (idx 6)
    (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (4, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (1, 5), # K2 to K1 (idx 19)
    (1, 6), (1, 7), (1, 8) # K1 to Door (idx 22)
]

# 3. Generate frames
frames = []
for i in range(63):
    t = i * (22.0 / 62)
    
    # Calculate agent pos
    segment = int(t)
    if segment >= 22:
        r, c = full_path[-1]
    else:
        t_rem = t - segment
        r1, c1 = full_path[segment]
        r2, c2 = full_path[segment+1]
        r = r1 + (r2 - r1) * t_rem
        c = c1 + (c2 - c1) * t_rem
    
    agent_y, agent_x = grid_to_pixel(r, c)
    
    # Create frame
    frame = bg.copy()
    
    # Draw K2 if t < 6.0
    if t < 6.0:
        for dy, dx in k2_shape:
            y, x = k2_center[0] + dy, k2_center[1] + dx
            frame[y, x] = [255, 128, 0]
            
    # Draw K1 if t < 19.0
    if t < 19.0:
        for dy, dx in k1_shape:
            y, x = k1_center[0] + dy, k1_center[1] + dx
            frame[y, x] = [0, 255, 255]
            
    # Draw agent
    for dy, dx in agent_shape:
        y, x = agent_y + dy, agent_x + dx
        if 0 <= y < 1024 and 0 <= x < 1024:
            frame[y, x] = [0, 255, 0]
            
    frames.append(frame)

# 4. Save video
os.makedirs('/app/output', exist_ok=True)
imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')
