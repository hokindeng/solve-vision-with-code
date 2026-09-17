import cv2
import numpy as np
import imageio
import os
import base64

# 1. Read image and convert to RGB
img_bgr = cv2.imread('/app/first_frame.png')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Extract patches 1, 3, 4
patch_4 = img[274:274+127, 449:449+127].copy()
patch_3 = img[566:566+127, 741:741+127].copy()
patch_1 = img[566:566+127, 157:157+127].copy()

# 3. Load patch 2 from base64
patch_2_b64 = "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"
patch_2_bytes = base64.b64decode(patch_2_b64)
patch_2_np = np.frombuffer(patch_2_bytes, np.uint8)
patch_2_bgr = cv2.imdecode(patch_2_np, cv2.IMREAD_COLOR)
patch_2 = cv2.cvtColor(patch_2_bgr, cv2.COLOR_BGR2RGB)

patches = {
    1: patch_1,
    2: patch_2,
    3: patch_3,
    4: patch_4
}

# 4. Define bounding boxes
lights = {
    'North': {'light': (447, 155, 131, 119), 'text': (449, 274, 127, 127)},
    'South': {'light': (447, 739, 131, 119), 'text': (449, 858, 127, 127)},
    'East':  {'light': (739, 447, 131, 119), 'text': (741, 566, 127, 127)},
    'West':  {'light': (155, 447, 131, 119), 'text': (157, 566, 127, 127)}
}

colors_rgb = {
    'Red': [255, 0, 0],
    'Yellow': [255, 200, 0],
    'Green': [0, 200, 0]
}

# cycle definition:
cycle = [
    ('Red', 4), ('Red', 3), ('Red', 2), ('Red', 1),
    ('Yellow', 4), ('Yellow', 3), ('Yellow', 2), ('Yellow', 1),
    ('Green', 4), ('Green', 3), ('Green', 2), ('Green', 1),
    ('Yellow', 4), ('Yellow', 3), ('Yellow', 2), ('Yellow', 1)
]

state_indices = {
    'North': 0,
    'South': 8,
    'East': 9,
    'West': 15
}

os.makedirs('/app/output', exist_ok=True)
writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None, quality=10)

for frame_idx in range(112): # 0 to 111
    # Pace the action over the full duration rather than jumping to the end.
    t = frame_idx // 16 
    
    frame = img.copy()
    
    for direction in ['North', 'South', 'East', 'West']:
        current_idx = (state_indices[direction] + t) % 16
        color_name, digit = cycle[current_idx]
        
        # update text
        tx, ty, tw, th = lights[direction]['text']
        frame[ty:ty+th, tx:tx+tw] = patches[digit]
        
        # update light
        lx, ly, lw, lh = lights[direction]['light']
        light_patch = frame[ly:ly+lh, lx:lx+lw]
        
        mask_black = (light_patch[:,:,0] == 0) & (light_patch[:,:,1] == 0) & (light_patch[:,:,2] == 0)
        mask_gray = (light_patch[:,:,0] == 60) & (light_patch[:,:,1] == 60) & (light_patch[:,:,2] == 60)
        mask_colored = ~(mask_black | mask_gray)
        
        light_patch[mask_colored] = colors_rgb[color_name]
        
    writer.append_data(frame)

writer.close()
