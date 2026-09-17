import numpy as np
import imageio.v2 as iio
from collections import deque
import os
import subprocess
import shutil

def get_grid_centers():
    grid_size = 11
    cell_size = 867 / 11.0
    centers = {}
    for i in range(grid_size):
        for j in range(grid_size):
            cy = int(78 + i * cell_size + cell_size / 2)
            cx = int(78 + j * cell_size + cell_size / 2)
            centers[(i, j)] = (cy, cx)
    return centers

def get_path():
    grid = [
        "WWWWWWWBWWW",
        "BBBBBBWBBBW",
        "WWWWUBWWWBW",
        "WBBBWBBBWBW",
        "WBWWWBWWWBW",
        "CBWBBBWBBBW",
        "WBWWWBWWWGW",
        "WBBBBBBBBBW",
        "WWRWWWWWWBW",
        "WBBBBBWBBBW",
        "WWWWWBWWWWW",
    ]

    nodes = {'G': (6,9), 'C': (5,0), 'R': (8,2), 'U': (2,4)}

    def bfs_path(start, end):
        q = deque([(start, [start])])
        visited = set([start])
        while q:
            (r, c), path = q.popleft()
            if (r, c) == end:
                return path
            for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                if 0 <= nr < 11 and 0 <= nc < 11 and grid[nr][nc] != 'B':
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append(((nr, nc), path + [(nr, nc)]))
        return []

    path_G_R = bfs_path(nodes['G'], nodes['R'])
    path_R_C = bfs_path(nodes['R'], nodes['C'])
    path_C_U = bfs_path(nodes['C'], nodes['U'])

    full_path = path_G_R + path_R_C[1:] + path_C_U[1:]
    return full_path

def main():
    first_frame = iio.imread('/app/first_frame.png')
    
    green_mask = np.all(first_frame == [0, 255, 0], axis=-1)
    red_mask = np.all(first_frame == [255, 0, 0], axis=-1)
    cyan_mask = np.all(first_frame == [0, 255, 255], axis=-1)
    
    ay, ax = np.where(green_mask)
    cy_agent, cx_agent = 590, 826
    ay_rel = ay - cy_agent
    ax_rel = ax - cx_agent
    
    base_img = first_frame.copy()
    base_img[green_mask] = [255, 255, 255]
    
    full_path = get_path()
    centers = get_grid_centers()
    
    tmp_dir = '/tmp/maze_frames'
    os.makedirs(tmp_dir, exist_ok=True)
    
    for f in range(73):
        progress = (f / 72.0) * 27.0
        
        # update base_img if we passed keys
        if progress >= 15.0:
            base_img[red_mask] = [255, 255, 255]
        if progress >= 20.0:
            base_img[cyan_mask] = [255, 255, 255]
            
        frame = base_img.copy()
        
        edge_idx = int(np.floor(progress))
        if edge_idx >= 27:
            edge_idx = 26
            t = 1.0
        else:
            t = progress - edge_idx
            
        node1 = full_path[edge_idx]
        node2 = full_path[edge_idx + 1]
        
        cy1, cx1 = centers[node1]
        cy2, cx2 = centers[node2]
        
        cy = int(cy1 + t * (cy2 - cy1))
        cx = int(cx1 + t * (cx2 - cx1))
        
        frame[ay_rel + cy, ax_rel + cx] = [0, 255, 0]
        
        iio.imwrite(f'{tmp_dir}/frame_{f:04d}.png', frame)
        
    os.makedirs('/app/output', exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', 
        '-i', f'{tmp_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    main()
