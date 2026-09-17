import imageio as iio
import numpy as np
import cv2

first_frame = cv2.imread('/app/first_frame.png')
# Convert BGR to RGB for imageio
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

bg = first_frame.copy()
for r in range(3):
    for c in range(3):
        y = 36 + r * 328
        x = 36 + c * 328
        bg[y:y+296, x:x+296] = (255, 255, 255)

start_state = [4, 0, 2, 8, 1, 3, 5, 7, 6]
moves = [1, 8, 5, 7, 8, 5, 4, 1, 2, 3, 6]

tiles = {}
for idx, tile_id in enumerate(start_state):
    if tile_id == 0: continue
    r, c = divmod(idx, 3)
    y = 36 + r * 328
    x = 36 + c * 328
    tiles[tile_id] = first_frame[y:y+296, x:x+296].copy()

animations = []
current_state = list(start_state)
for tile_id in moves:
    src_idx = current_state.index(tile_id)
    dst_idx = current_state.index(0)
    
    animations.append({
        'tile_id': tile_id,
        'src_idx': src_idx,
        'dst_idx': dst_idx,
        'state_before': list(current_state)
    })
    
    current_state[src_idx], current_state[dst_idx] = current_state[dst_idx], current_state[src_idx]

total_frames = 76
num_moves = len(moves)
writer = iio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

for i in range(total_frames):
    t = i * num_moves / (total_frames - 1)
    move_idx = min(int(t), num_moves - 1)
    p = t - move_idx
    if move_idx == num_moves - 1 and p > 1.0:
        p = 1.0
        
    ease_p = p * p * (3 - 2 * p)
    
    anim = animations[move_idx]
    state = anim['state_before']
    
    frame = bg.copy()
    
    for idx, tile_id in enumerate(state):
        if tile_id == 0 or tile_id == anim['tile_id']:
            continue
        r, c = divmod(idx, 3)
        y = 36 + r * 328
        x = 36 + c * 328
        frame[y:y+296, x:x+296] = tiles[tile_id]
        
    src_r, src_c = divmod(anim['src_idx'], 3)
    dst_r, dst_c = divmod(anim['dst_idx'], 3)
    
    src_y = 36 + src_r * 328
    src_x = 36 + src_c * 328
    dst_y = 36 + dst_r * 328
    dst_x = 36 + dst_c * 328
    
    curr_y = int(src_y + (dst_y - src_y) * ease_p)
    curr_x = int(src_x + (dst_x - src_x) * ease_p)
    
    frame[curr_y:curr_y+296, curr_x:curr_x+296] = tiles[anim['tile_id']]
    
    writer.append_data(frame)

writer.close()
