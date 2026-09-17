import cv2
import numpy as np
import os
import subprocess

# Ensure output dir exists
os.makedirs('/app/output', exist_ok=True)

first_frame = cv2.imread('/app/first_frame.png')
if first_frame is None:
    raise ValueError("Could not read /app/first_frame.png")

# Initial grid and solution
initial_grid = [8, 0, 2, 1, 4, 3, 7, 6, 5]
moves = [0, 3, 4, 1, 2, 5, 8, 7, 4, 5, 8]

# Compute states
states = [list(initial_grid)]
move_from = []
move_to = []

curr = list(initial_grid)
for m in moves:
    empty_idx = curr.index(0)
    move_from.append(m)
    move_to.append(empty_idx)
    curr[empty_idx], curr[m] = curr[m], curr[empty_idx]
    states.append(list(curr))

# Extract tiles
tile_images = {}
background = first_frame.copy()

for idx, val in enumerate(initial_grid):
    row = idx // 3
    col = idx % 3
    x = 36 + 328 * col
    y = 36 + 328 * row
    
    if val != 0:
        tile_images[val] = first_frame[y:y+296, x:x+296].copy()
    
    # Erase the area in the background
    background[y:y+296, x:x+296] = 255

# Generate frames
temp_video = '/tmp/temp.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_video, fourcc, 16.0, (1024, 1024))

def ease(t):
    return t * t * (3 - 2 * t)

TOTAL_FRAMES = 76
for f in range(TOTAL_FRAMES):
    if f < 5:
        grid = states[0]
        moving_tile_val = None
    elif f >= 71:
        grid = states[-1]
        moving_tile_val = None
    else:
        f_active = f - 5
        m = f_active // 6
        fraction = (f_active % 6) / 6.0
        grid = states[m]
        moving_tile_val = grid[move_from[m]]
        start_idx = move_from[m]
        end_idx = move_to[m]
        
    img = background.copy()
    
    # Draw static tiles
    for idx in range(9):
        val = grid[idx]
        if val == 0: continue
        if val == moving_tile_val: continue
        
        row = idx // 3
        col = idx % 3
        x = 36 + 328 * col
        y = 36 + 328 * row
        img[y:y+296, x:x+296] = tile_images[val]
        
    # Draw moving tile
    if moving_tile_val is not None:
        sr = start_idx // 3
        sc = start_idx % 3
        er = end_idx // 3
        ec = end_idx % 3
        
        sx = 36 + 328 * sc
        sy = 36 + 328 * sr
        ex = 36 + 328 * ec
        ey = 36 + 328 * er
        
        t = ease(fraction)
        x = int(sx + (ex - sx) * t)
        y = int(sy + (ey - sy) * t)
        
        img[y:y+296, x:x+296] = tile_images[moving_tile_val]
        
    out.write(img)

out.release()

# Convert to H.264, yuv420p
subprocess.run([
    'ffmpeg', '-y', '-i', temp_video,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '16',
    '/app/output/video.mp4'
], check=True)

