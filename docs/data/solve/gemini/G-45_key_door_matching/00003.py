import cv2
import numpy as np
import os
import subprocess

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Get agent mask and coordinates
    green_mask = cv2.inRange(img, np.array([0, 255, 0]), np.array([0, 255, 0]))
    agent_pixels = np.where(green_mask > 0)
    
    # Center of the green agent in the original image
    min_y, max_y = agent_pixels[0].min(), agent_pixels[0].max()
    min_x, max_x = agent_pixels[1].min(), agent_pixels[1].max()
    orig_cy = (min_y + max_y) / 2.0
    orig_cx = (min_x + max_x) / 2.0
    
    # 2. Prepare background with key (agent erased)
    bg_with_key = img.copy()
    bg_with_key[green_mask > 0] = [255, 255, 255]
    
    # 3. Prepare background without key (agent erased, key erased)
    bg_without_key = bg_with_key.copy()
    cell_size = 1024 / 15
    y1 = int(round(2 * cell_size))
    y2 = int(round(3 * cell_size))
    x1 = int(round(13 * cell_size))
    x2 = int(round(14 * cell_size))
    
    cell_img = bg_without_key[y1:y2, x1:x2]
    yellow_mask = cv2.inRange(cell_img, np.array([0, 255, 255]), np.array([0, 255, 255]))
    cell_img[yellow_mask > 0] = [255, 255, 255]
    bg_without_key[y1:y2, x1:x2] = cell_img
    
    # Calculate offset so the agent at (1, 1) perfectly matches orig_cx, orig_cy
    offset_x = orig_cx - (1 + 0.5) * cell_size
    offset_y = orig_cy - (1 + 0.5) * cell_size
    
    # 4. Generate path
    grid_str = """
###############
#...#...#.....#
#.#.#.#.#.###.#
#.#...#...#...#
###########.#.#
#...........#.#
#.###########.#
#.#.#.......#.#
#.#.#.#######.#
#.#.#.#...#...#
#.#.#.#.#.#.###
#...#.#.#.#.#.#
#####.#.#.#.#.#
#.......#.....#
###############
"""
    grid = [list(line.strip()) for line in grid_str.strip().split('\n')]
    
    def bfs(start, end):
        queue = [[start]]
        visited = set([start])
        while queue:
            path = queue.pop(0)
            r, c = path[-1]
            if (r, c) == end:
                return path
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < 15 and 0 <= nc < 15 and grid[nr][nc] == '.' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
        return None

    path1 = bfs((1, 1), (2, 13))
    path2 = bfs((2, 13), (9, 9))
    full_path = path1 + path2[1:]
    
    # 5. Generate frames
    os.makedirs('/app/frames', exist_ok=True)
    num_frames = 94
    total_steps = len(full_path) - 1
    
    for i in range(num_frames):
        progress = i * total_steps / (num_frames - 1)
        idx = int(progress)
        frac = progress - idx
        
        if idx >= total_steps:
            idx = total_steps - 1
            frac = 1.0
            if i == num_frames - 1:
                idx = total_steps
                frac = 0.0
        
        if idx < total_steps:
            r1, c1 = full_path[idx]
            r2, c2 = full_path[idx + 1]
            r = r1 + (r2 - r1) * frac
            c = c1 + (c2 - c1) * frac
        else:
            r, c = full_path[-1]
        
        # Decide which background to use
        # Key disappears when agent reaches it (index 21)
        if progress >= 21.0:
            frame = bg_without_key.copy()
        else:
            frame = bg_with_key.copy()
            
        # Draw agent
        new_cx = (c + 0.5) * cell_size + offset_x
        new_cy = (r + 0.5) * cell_size + offset_y
        
        ys = np.round(agent_pixels[0] - orig_cy + new_cy).astype(int)
        xs = np.round(agent_pixels[1] - orig_cx + new_cx).astype(int)
        
        # Clip just in case
        valid = (ys >= 0) & (ys < 1024) & (xs >= 0) & (xs < 1024)
        ys = ys[valid]
        xs = xs[valid]
        
        frame[ys, xs] = [0, 255, 0]
        
        # Exact first frame enforcement to guarantee exactly identical
        if i == 0:
            frame = img.copy()
            
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)
        
    # 6. Encode video
    os.makedirs('/app/output', exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    create_video()
