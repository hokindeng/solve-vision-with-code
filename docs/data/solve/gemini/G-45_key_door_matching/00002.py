import cv2
import numpy as np
import imageio
from collections import deque
import sys
import os

os.makedirs('/app/output', exist_ok=True)

# Load image
img = cv2.imread('/app/first_frame.png')
if img is None:
    print("Could not read first_frame.png")
    sys.exit(1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

offset = 53
cell_size = 54
grid_size = 17

maze = np.zeros((grid_size, grid_size), dtype=int)
for r in range(grid_size):
    for c in range(grid_size):
        cy = offset + r * cell_size
        cx = offset + c * cell_size
        cell = img[cy:cy+cell_size, cx:cx+cell_size]
        
        if np.any(np.all(cell == [0, 255, 0], axis=-1)):
            maze[r, c] = 2
        elif np.all(cell[cell_size//2, cell_size//2] == [0, 0, 0]):
            maze[r, c] = 0
        else:
            maze[r, c] = 1

def bfs(start, goal):
    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft()
        r, c = path[-1]
        if (r, c) == goal:
            return path
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and maze[nr][nc] != 0:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append(path + [(nr, nc)])
    return None

path1 = bfs((0,0), (10,4))
path2 = bfs((10,4), (0,10))

if not path1 or not path2:
    print("Path not found!")
    sys.exit(1)

frames_coords = []
for _ in range(4):
    frames_coords.append(path1[0])

for i in range(len(path1) - 1):
    r1, c1 = path1[i]
    r2, c2 = path1[i+1]
    frames_coords.append( (r1 + (r2 - r1)/2, c1 + (c2 - c1)/2) )
    frames_coords.append( (r2, c2) )

key_collected_idx = len(frames_coords) - 1

for _ in range(6):
    frames_coords.append(path1[-1])

for i in range(len(path2) - 1):
    r1, c1 = path2[i]
    r2, c2 = path2[i+1]
    frames_coords.append( (r1 + (r2 - r1)/2, c1 + (c2 - c1)/2) )
    frames_coords.append( (r2, c2) )

for _ in range(4):
    frames_coords.append(path2[-1])

print(f"Total frames: {len(frames_coords)}")

# Prepare masks and backgrounds
agent_patch = img[offset:offset+cell_size, offset:offset+cell_size].copy()
agent_mask = (agent_patch == [0, 255, 0]).all(axis=-1)

bg_img = img.copy()
bg_img[offset:offset+cell_size, offset:offset+cell_size][agent_mask] = [255, 255, 255]

key_r, key_c = 10, 4
ky1 = offset + key_r * cell_size
ky2 = ky1 + cell_size
kx1 = offset + key_c * cell_size
kx2 = kx1 + cell_size

key_patch = bg_img[ky1:ky2, kx1:kx2].copy()
red_mask = (key_patch == [255, 0, 0]).all(axis=-1) # Note: RGB, red is [255, 0, 0]

bg_img_no_key = bg_img.copy()
bg_img_no_key[ky1:ky2, kx1:kx2][red_mask] = [255, 255, 255]

writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

for i, (r, c) in enumerate(frames_coords):
    if i >= key_collected_idx:
        frame = bg_img_no_key.copy()
    else:
        frame = bg_img.copy()
        
    y = int(offset + r * cell_size)
    x = int(offset + c * cell_size)
    
    frame[y:y+cell_size, x:x+cell_size][agent_mask] = [0, 255, 0]
    
    writer.append_data(frame)

writer.close()
print("Video generated successfully.")
