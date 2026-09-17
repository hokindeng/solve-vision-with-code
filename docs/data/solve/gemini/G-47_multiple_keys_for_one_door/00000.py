import cv2
import numpy as np
import imageio
import os

# 1. Load the first frame
first_frame = cv2.imread('/app/first_frame.png')
# OpenCV loads in BGR. imageio expects RGB.
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# 2. Extract agent points and define background
agent_color = np.array([0, 255, 0])
mask_agent = cv2.inRange(first_frame_rgb, agent_color, agent_color)
y_agent, x_agent = np.where(mask_agent == 255)
# Use exact measured center of the agent
cx_agent, cy_agent = 605, 139
agent_offsets = np.column_stack((x_agent - cx_agent, y_agent - cy_agent))

# We create a base background with no agent
bg_base = first_frame_rgb.copy()
bg_base[mask_agent == 255] = [255, 255, 255]

# Identify keys to erase them later
blue_key = np.array([0, 0, 255])     # RGB for Blue
orange_key = np.array([255, 128, 0]) # RGB for Orange

mask_k1 = cv2.inRange(bg_base, blue_key, blue_key)
mask_k2 = cv2.inRange(bg_base, orange_key, orange_key)

# 3. Path planning (calculated previously)
path = [
    (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 8), (2, 8), (2, 7), (2, 6),
    (3, 6), (4, 6), (4, 7), (4, 8),
    (5, 8), (6, 8), (7, 8), (8, 8),
    (8, 7)
]

def r_c_to_x_y(r, c):
    # exact overrides for perfectly centering on objects
    if (r, c) == (0, 5): return 605, 139 # Agent Start
    if (r, c) == (4, 7): return 791, 511 # Key 2 (Orange)
    if (r, c) == (8, 8): return 884, 884 # Key 1 (Blue)
    if (r, c) == (8, 7): return 791, 884 # Door (Magenta)
    
    cell_size = 93
    start_x = 93
    start_y = 93
    cx = start_x + c * cell_size + 46
    cy = start_y + r * cell_size + 46
    return cx, cy

path_xy = [r_c_to_x_y(r, c) for r, c in path]

# 4. Generate frames
total_frames = 51
frames = []

for i in range(total_frames):
    # Progress along the path [0, 16]
    dist = i * 16.0 / (total_frames - 1)
    
    # State of the background
    current_bg = bg_base.copy()
    
    if dist >= 10.0:
        # Erase K2 (Orange)
        current_bg[mask_k2 == 255] = [255, 255, 255]
    if dist >= 15.0:
        # Erase K1 (Blue)
        current_bg[mask_k1 == 255] = [255, 255, 255]
        
    # Interpolate agent position
    idx = int(dist)
    if idx >= 16:
        idx = 15
        frac = 1.0
    else:
        frac = dist - idx
        
    p0 = path_xy[idx]
    p1 = path_xy[idx+1]
    
    curr_x = int(round(p0[0] + frac * (p1[0] - p0[0])))
    curr_y = int(round(p0[1] + frac * (p1[1] - p0[1])))
    
    # Draw agent
    frame = current_bg.copy()
    
    # Agent mask mapping
    ax = agent_offsets[:, 0] + curr_x
    ay = agent_offsets[:, 1] + curr_y
    
    # Clip just in case (though it shouldn't go out of bounds)
    valid = (ax >= 0) & (ax < 1024) & (ay >= 0) & (ay < 1024)
    ax = ax[valid]
    ay = ay[valid]
    
    frame[ay, ax] = [0, 255, 0]
    
    if i == 0:
        frame = first_frame_rgb.copy()
        
    frames.append(frame)

os.makedirs('/app/output', exist_ok=True)
imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

