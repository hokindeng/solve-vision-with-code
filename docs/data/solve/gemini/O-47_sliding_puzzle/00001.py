import cv2
import numpy as np
import imageio
import math
import os

os.makedirs('/app/output', exist_ok=True)

# 1. Load the first frame
img = cv2.imread('/app/first_frame.png')

# 2. Define grid lines and grid mask
grid_mask_fixed = np.zeros(img.shape[:2], dtype=bool)
lines = [20, 21, 348, 349, 675, 676, 1003, 1004]
for l in lines:
    grid_mask_fixed[l, 20:1005] = True
    grid_mask_fixed[20:1005, l] = True

grid_mask_original = (img == [51, 51, 51]).all(axis=-1)
final_grid_mask = grid_mask_fixed & grid_mask_original

# 3. Define board state and moves
slot_coords = {
    0: (36, 36),   1: (36, 364),  2: (36, 692),
    3: (364, 36),  4: (364, 364), 5: (364, 692),
    6: (692, 36),  7: (692, 364), 8: (692, 692)
}

moves = [
    (8, 5), (7, 8), (4, 7), (5, 4),
    (2, 5), (1, 2), (0, 1), (3, 0),
    (4, 3), (7, 4), (8, 7)
]

frames_per_move = [7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6]

board = img.copy()

# Initialize video writer with precise h264 and yuv420p
writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1)
# Add first frame
writer.append_data(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))

for move_idx, (m_from, m_to) in enumerate(moves):
    y_from, x_from = slot_coords[m_from]
    y_to, x_to = slot_coords[m_to]
    
    # Extract tile
    tile = board[y_from:y_from+296, x_from:x_from+296].copy()
    
    # Clear tile from board
    board[y_from:y_from+296, x_from:x_from+296] = 255
    
    N = frames_per_move[move_idx]
    
    for i in range(1, N + 1):
        t = i / N
        # Linear easing to maintain constant speed?
        # Actually smooth easing looks better. The prompt just says "pace the action over the full duration"
        t = (1 - math.cos(t * math.pi)) / 2
        
        cur_y = int(round(y_from + (y_to - y_from) * t))
        cur_x = int(round(x_from + (x_to - x_from) * t))
        
        frame = board.copy()
        # Paste tile
        frame[cur_y:cur_y+296, cur_x:cur_x+296] = tile
        
        # Restore static grid lines
        frame[final_grid_mask] = [51, 51, 51]
        
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    # Update board for next move
    board[y_to:y_to+296, x_to:x_to+296] = tile
    board[final_grid_mask] = [51, 51, 51]

writer.close()
