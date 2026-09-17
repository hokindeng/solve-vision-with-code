import cv2
import numpy as np
import imageio

# 1. Load original image
img = cv2.imread('/app/first_frame.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Extract agent patch and mask
green_mask = cv2.inRange(img_rgb, np.array([0, 255, 0]), np.array([0, 255, 0]))
agent_patch = img_rgb[882:930, 882:930].copy()
agent_mask = green_mask[882:930, 882:930] > 0

# 3. Create backgrounds
bg_all_keys = img_rgb.copy()
bg_all_keys[green_mask > 0] = [255, 255, 255]

blue_mask = cv2.inRange(bg_all_keys, np.array([0, 0, 255]), np.array([0, 0, 255])) # Blue in RGB is [0, 0, 255]
bg_magenta_key = bg_all_keys.copy()
bg_magenta_key[blue_mask > 0] = [255, 255, 255]

magenta_mask = cv2.inRange(bg_all_keys, np.array([255, 0, 255]), np.array([255, 0, 255])) # Magenta in RGB is [255, 0, 255]
bg_no_keys = bg_magenta_key.copy()
bg_no_keys[magenta_mask > 0] = [255, 255, 255]

def get_agent_tl(c, r):
    x = int(round(882 + (c - 11) * (1024 / 13.0)))
    y = int(round(882 + (r - 11) * (1024 / 13.0)))
    return x, y

def paste_agent(bg, tl_x, tl_y):
    out = bg.copy()
    patch_bg = out[tl_y:tl_y+48, tl_x:tl_x+48]
    patch_bg[agent_mask] = agent_patch[agent_mask]
    out[tl_y:tl_y+48, tl_x:tl_x+48] = patch_bg
    return out

path_grid = [
    (11, 11), (11, 10), (11, 9), (10, 9), (9, 9), 
    (9, 10), (9, 11), (8, 11), (7, 11), (6, 11), 
    (5, 11), (4, 11), (3, 11), (2, 11), (1, 11), (1, 10)
]

frames = []
total_frames = 49
num_segments = len(path_grid) - 1

for i in range(total_frames):
    if i == 0:
        # Exactly first frame
        frames.append(img_rgb)
        continue
        
    progress = i * num_segments / (total_frames - 1)
    
    if progress < 5.0:
        bg = bg_all_keys
    elif progress < 10.0:
        bg = bg_magenta_key
    else:
        bg = bg_no_keys
        
    seg_idx = int(progress)
    if seg_idx >= num_segments:
        seg_idx = num_segments - 1
        t = 1.0
    else:
        t = progress - seg_idx
        
    r1, c1 = path_grid[seg_idx]
    r2, c2 = path_grid[seg_idx + 1]
    
    r_curr = r1 + (r2 - r1) * t
    c_curr = c1 + (c2 - c1) * t
    
    tl_x, tl_y = get_agent_tl(c_curr, r_curr)
    
    frame = paste_agent(bg, tl_x, tl_y)
    frames.append(frame)

# Save video
writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
for frame in frames:
    writer.append_data(frame)
writer.close()
