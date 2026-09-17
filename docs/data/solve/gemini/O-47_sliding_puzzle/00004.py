import cv2
import numpy as np
from collections import deque
import subprocess
import os
import shutil

def solve():
    start = (2, 5, 3, 4, 6, 0, 7, 1, 8)
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def get_neighbors(state):
        idx = state.index(0)
        row, col = divmod(idx, 3)
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                nidx = nr * 3 + nc
                new_state = list(state)
                new_state[idx], new_state[nidx] = new_state[nidx], new_state[idx]
                neighbors.append((tuple(new_state), state[nidx], nidx, idx))
        return neighbors

    q = deque([(start, [])])
    visited = {start}
    path = None

    while q:
        curr, p = q.popleft()
        if curr == goal:
            path = p
            break
        
        for nxt_state, tile_moved, from_idx, to_idx in get_neighbors(curr):
            if nxt_state not in visited:
                visited.add(nxt_state)
                q.append((nxt_state, p + [(tile_moved, from_idx, to_idx, nxt_state)]))
    
    return start, path

def main():
    start, path = solve()
    
    img = cv2.imread('/app/first_frame.png')
    rects = [
        (36, 36), (364, 36), (692, 36),
        (36, 364), (364, 364), (692, 364),
        (36, 692), (364, 692), (692, 692)
    ]
    
    tiles = {}
    for i, val in enumerate(start):
        if val != 0:
            x, y = rects[i]
            tiles[val] = img[y:y+296, x:x+296].copy()
            
    # Create background by clearing the 9 cells
    bg = img.copy()
    for x, y in rects:
        bg[y:y+296, x:x+296] = 255
        
    # State sequence: list of dicts mapping tile_val -> cell_idx
    state_seq = []
    curr_state_dict = {val: i for i, val in enumerate(start) if val != 0}
    state_seq.append(curr_state_dict.copy())
    
    for step in path:
        tile_moved = step[0]
        to_idx = step[2]
        curr_state_dict[tile_moved] = to_idx
        state_seq.append(curr_state_dict.copy())
        
    num_frames = 76
    
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/frames', exist_ok=True)
    
    for f in range(num_frames):
        t = f * 11.0 / (num_frames - 1)
        move_idx = int(t)
        progress = t - move_idx
        
        if move_idx >= 11:
            move_idx = 10
            progress = 1.0
            
        frame = bg.copy()
        
        state_start = state_seq[move_idx]
        state_end = state_seq[move_idx + 1]
        
        for val in range(1, 9):
            idx_start = state_start[val]
            idx_end = state_end[val]
            
            x_start, y_start = rects[idx_start]
            x_end, y_end = rects[idx_end]
            
            x = int(round(x_start + progress * (x_end - x_start)))
            y = int(round(y_start + progress * (y_end - y_start)))
            
            frame[y:y+296, x:x+296] = tiles[val]
            
        cv2.imwrite(f'/app/frames/frame_{f:04d}.png', frame)
        
    # Convert frames to standard H.264 yuv420p video
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ], check=True)
    
    # cleanup frames
    shutil.rmtree('/app/frames')

if __name__ == '__main__':
    main()
