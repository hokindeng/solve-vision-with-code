import cv2
import numpy as np
import math
import shutil
import os
import subprocess
from collections import deque

def bfs(grid, start, goal):
    q = deque([(start, [start])])
    visited = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == '.':
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
    return None

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Based on our analysis, the grid is 13x13 and cell size is 1024/13.
    grid_str = [
        "# # # # # # # # # # # # #",
        "# . . . . . . . . . . . #",
        "# . # # # # # . # # # . #",
        "# . # . # . # . # . . . #",
        "# . # . # . # . # . # . #",
        "# . . . # . # . # . # . #",
        "# # # # # . # . # . # . #",
        "# . . . . . # . # . # . #",
        "# . # # # # # . # . # # #",
        "# . # . . . # . # . . . #",
        "# . # . # . # # # # # . #",
        "# . . . # . . . . . . . #",
        "# # # # # # # # # # # # #"
    ]
    grid = [row.split() for row in grid_str]
    
    # Detected locations from first_frame.png
    agent = (1, 2)
    keys = [(2, 7), (5, 2)]
    door = (4, 3)
    
    # Calculate shortest path in optimal order (Key 0 -> Key 1 -> Door)
    p1 = bfs(grid, agent, keys[0])
    p2 = bfs(grid, keys[0], keys[1])
    p3 = bfs(grid, keys[1], door)
    full_path = p1 + p2[1:] + p3[1:]
    
    total_steps = len(full_path) - 1
    
    def get_pixel_pos(r, c):
        return (c + 0.5) * 1024.0 / 13.0, (r + 0.5) * 1024.0 / 13.0
    
    path_pixels = [get_pixel_pos(r, c) for r, c in full_path]
    
    # Prepare a clean background without the agent
    bg_clean = img.copy()
    agent_color = [0, 255, 0]
    green_mask = np.all(bg_clean == agent_color, axis=-1)
    bg_clean[green_mask] = [255, 255, 255]
    
    # Extract agent sprite for drawing
    agent_mask = np.all(img == agent_color, axis=-1)
    agent_y, agent_x = np.where(agent_mask)
    ay_min, ay_max = agent_y.min(), agent_y.max()
    ax_min, ax_max = agent_x.min(), agent_x.max()
    
    agent_sprite = img[ay_min:ay_max+1, ax_min:ax_max+1].copy()
    agent_sprite_mask = agent_mask[ay_min:ay_max+1, ax_min:ax_max+1]
    
    os.makedirs('/tmp/frames', exist_ok=True)
    
    # First frame exactly as original
    shutil.copy('/app/first_frame.png', '/tmp/frames/0000.png')
    
    total_frames = 59
    duration_frames = total_frames - 1 # 58 frames
    
    # Find indices for keys along the path
    key0_idx = full_path.index(keys[0])
    key1_idx = full_path.index(keys[1])
    
    # Generate animation frames
    for t in range(1, total_frames):
        d = t * total_steps / float(duration_frames)
        idx = int(math.floor(d))
        frac = d - idx
        
        if idx >= total_steps:
            cx, cy = path_pixels[-1]
        else:
            cx0, cy0 = path_pixels[idx]
            cx1, cy1 = path_pixels[idx+1]
            cx = cx0 + frac * (cx1 - cx0)
            cy = cy0 + frac * (cy1 - cy0)
            
        frame = bg_clean.copy()
        
        # Remove keys when reached
        if d >= key0_idx:
            key0_mask = np.all(frame == [255, 0, 0], axis=-1)
            frame[key0_mask] = [255, 255, 255]
        if d >= key1_idx:
            key1_mask = np.all(frame == [255, 0, 255], axis=-1)
            frame[key1_mask] = [255, 255, 255]
            
        # Draw agent
        x1 = int(round(cx)) - 24
        y1 = int(round(cy)) - 24
        
        h, w = agent_sprite.shape[:2]
        frame[y1:y1+h, x1:x1+w][agent_sprite_mask] = agent_sprite[agent_sprite_mask]
        
        cv2.imwrite(f'/tmp/frames/{t:04d}.png', frame)
        
    os.makedirs('/app/output', exist_ok=True)
    
    # Encode with FFmpeg
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/tmp/frames/%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
