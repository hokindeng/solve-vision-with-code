import cv2
import numpy as np
import imageio
from collections import deque
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # Initial state identified from the image
    initial_state = (5, 1, 3, 0, 7, 6, 2, 4, 8)
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    
    # Find shortest path using BFS
    queue = deque([(initial_state, [])])
    visited = set([initial_state])
    path = []
    while queue:
        state, p = queue.popleft()
        if state == goal_state:
            path = p
            break
        zero_idx = state.index(0)
        row, col = zero_idx // 3, zero_idx % 3
        moves = []
        if row > 0: moves.append(-3) # Move zero up
        if row < 2: moves.append(3)  # Move zero down
        if col > 0: moves.append(-1) # Move zero left
        if col < 2: moves.append(1)  # Move zero right
        
        for move in moves:
            new_idx = zero_idx + move
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            new_state = tuple(new_state)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, p + [new_state[zero_idx]]))

    if not path:
        raise RuntimeError("No path found to solve the puzzle")

    # Extract the 296x296 tile graphics
    tiles = {}
    for i in range(9):
        val = initial_state[i]
        if val != 0:
            r = i // 3
            c = i % 3
            y = 36 + r * 328
            x = 36 + c * 328
            tiles[val] = img[y:y+296, x:x+296].copy()

    # Create a clean background by erasing the inner cell regions
    bg = img.copy()
    cell_bounds = [(22, 348), (350, 675), (677, 1003)]
    for r, (y1, y2) in enumerate(cell_bounds):
        for c, (x1, x2) in enumerate(cell_bounds):
            bg[y1:y2, x1:x2] = 255
            
    # The grid lines are dark gray (51, 51, 51)
    grid_mask = (bg == 51)

    # Frame 0 is the original image
    frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    current_state = list(initial_state)

    total_moves = len(path)
    total_generated_frames = 75 # Total animation frames

    for k in range(total_moves):
        # Distribute frames evenly over the moves
        start_f = round(k * total_generated_frames / total_moves)
        end_f = round((k + 1) * total_generated_frames / total_moves)
        
        moving_tile = path[k] 
        moving_idx = current_state.index(moving_tile)
        zero_idx = current_state.index(0)
        
        src_r, src_c = moving_idx // 3, moving_idx % 3
        dst_r, dst_c = zero_idx // 3, zero_idx % 3
        
        for f in range(start_f + 1, end_f + 1):
            p = (f - start_f) / (end_f - start_f)
            
            # Start with a pure white background
            frame = np.full((1024, 1024, 3), 255, dtype=np.uint8)
            
            # Draw all tiles
            for i in range(9):
                val = current_state[i]
                if val == 0: continue
                
                # Interpolate position for the moving tile
                if val == moving_tile:
                    curr_r = src_r + (dst_r - src_r) * p
                    curr_c = src_c + (dst_c - src_c) * p
                else:
                    curr_r = i // 3
                    curr_c = i % 3
                    
                y = round(36 + curr_r * 328)
                x = round(36 + curr_c * 328)
                
                frame[y:y+296, x:x+296] = tiles[val]
                
            # Overlay the static grid lines
            frame[grid_mask] = 51
            
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        # Update the state after the move is complete
        current_state[zero_idx], current_state[moving_idx] = current_state[moving_idx], current_state[zero_idx]

    os.makedirs('/app/output', exist_ok=True)
    video_path = '/app/output/video.mp4'
    imageio.mimwrite(video_path, frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
