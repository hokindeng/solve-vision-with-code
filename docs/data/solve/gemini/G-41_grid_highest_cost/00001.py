import cv2
import numpy as np
from PIL import Image
import os
import subprocess

# 1. Pathfinding
grid = [
    [0, 40, 50, 50],
    [10, 10, 30, 40],
    [40, 50, 30, 10],
    [10, 10, 40, 20]
]

best_cost = -1
best_path = []

def dfs(r, c, current_cost, path, visited):
    global best_cost, best_path
    if (r, c) == (3, 3):
        if current_cost > best_cost:
            best_cost = current_cost
            best_path = list(path)
        return
    
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4 and not visited[nr][nc]:
            visited[nr][nc] = True
            path.append((nr, nc))
            dfs(nr, nc, current_cost + grid[nr][nc], path, visited)
            path.pop()
            visited[nr][nc] = False

visited = [[False]*4 for _ in range(4)]
visited[0][0] = True
dfs(0, 0, 0, [(0,0)], visited)

# 2. Image Processing
img = cv2.imread('/app/first_frame.png')
H, W = img.shape[:2]

# Create clean background
clean = img.copy()
for i in range(256):
    for j in range(256):
        b, g, r = clean[i, j]
        if (b, g, r) != (200, 200, 200):
            clean[i, j] = (80, 175, 76)

clean_bg_pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))

# Extract Pac-Man sprite
sprite_rgba = np.zeros((128, 128, 4), dtype=np.uint8)
pacman_patch = img[84:212, 64:192]
for i in range(128):
    for j in range(128):
        b, g, r = pacman_patch[i, j]
        if (b, g, r) == (80, 175, 76) or (b, g, r) == (200, 200, 200):
            sprite_rgba[i, j] = (0, 0, 0, 0)
        else:
            dist_y = abs(int(r)-255) + abs(int(g)-255) + abs(int(b)-0)
            dist_b = abs(int(r)-0) + abs(int(g)-0) + abs(int(b)-0)
            dist_w = abs(int(r)-255) + abs(int(g)-255) + abs(int(b)-255)
            
            md = min(dist_y, dist_b, dist_w)
            if md == dist_y:
                sprite_rgba[i, j] = (255, 255, 0, 255) # Yellow
            elif md == dist_w:
                sprite_rgba[i, j] = (255, 255, 255, 255) # White
            else:
                sprite_rgba[i, j] = (0, 0, 0, 255) # Black

pacman_sprite_pil = Image.fromarray(sprite_rgba, 'RGBA')

# 3. Generate Frames
os.makedirs('/app/frames', exist_ok=True)
total_frames = 91
num_steps = len(best_path) - 1

for i in range(total_frames):
    if i == 0:
        frame_cv = img.copy()
    else:
        progress = i / (total_frames - 1)
        step_float = progress * num_steps
        step_index = int(step_float)
        
        if step_index >= num_steps:
            step_index = num_steps - 1
            t = 1.0
        else:
            t = step_float - step_index
            
        r1, c1 = best_path[step_index]
        r2, c2 = best_path[step_index+1]
        
        if r2 > r1: angle = -90
        elif r2 < r1: angle = 90
        elif c2 > c1: angle = 0
        elif c2 < c1: angle = 180
        else: angle = 0
        
        cx1 = c1 * 256 + 128
        cy1 = r1 * 256 + 148
        cx2 = c2 * 256 + 128
        cy2 = r2 * 256 + 148
        
        cx = int(round(cx1 + (cx2 - cx1) * t))
        cy = int(round(cy1 + (cy2 - cy1) * t))
        
        frame_pil = clean_bg_pil.copy()
        rotated_sprite = pacman_sprite_pil.rotate(angle)
        frame_pil.paste(rotated_sprite, (cx - 64, cy - 64), rotated_sprite)
        
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame_cv)

# 4. Encode Video
os.makedirs('/app/output', exist_ok=True)
subprocess.run([
    'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
], check=True)
