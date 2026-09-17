import cv2
import numpy as np
import imageio
import os

# Ensure output directory exists
os.makedirs('/app/output', exist_ok=True)

# Load first frame
first_frame = cv2.imread('/app/first_frame.png')

# Extract clean blocks based on the exact coordinates we found
# A: (219, 152, 52)   -> Right-0 bot
# B: (34, 126, 230)   -> Right-2 top
# C: (60, 76, 231)    -> Left-1 bot
# D: (182, 89, 155)   -> Left-2 top
# E: (15, 196, 241)   -> Right-1 bot
clean_blocks = {
    'A': first_frame[759:826, 591:690].copy(),
    'B': first_frame[631:698, 847:946].copy(),
    'C': first_frame[759:826, 207:306].copy(),
    'D': first_frame[695:762, 335:434].copy(),
    'E': first_frame[759:826, 719:818].copy()
}

# Create a background with left stacks erased
bg_erased = first_frame.copy()
bg_erased[100:828, 60:450] = (250, 245, 245)

def draw_block(img, blk_name, x, y):
    blk = clean_blocks[blk_name]
    img[y:y+67, x:x+99] = blk

# 8 moves to transform the stacks
moves = [
    (2, 0), # D from S2 to S0
    (2, 0), # E from S2 to S0
    (1, 2), # C from S1 to S2
    (0, 1), # E from S0 to S1
    (0, 2), # D from S0 to S2
    (0, 1), # A from S0 to S1
    (0, 2), # B from S0 to S2
    (1, 0)  # A from S1 to S0
]

def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)

# Initial state
stacks = [['B', 'A'], ['C'], ['E', 'D']]

frames = []
# Frame 0: exactly the first frame
frames.append(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

peak_y = 450

for move_idx, (src, dst) in enumerate(moves):
    blk = stacks[src].pop()
    
    start_x = 79 + src * 128
    start_y = 759 - len(stacks[src]) * 64
    
    end_x = 79 + dst * 128
    end_y = 759 - len(stacks[dst]) * 64
    
    for step in range(1, 19): # 18 frames per move
        t = step / 18.0
        t_eased = ease_in_out(t)
        
        D1 = start_y - peak_y
        D2 = abs(end_x - start_x)
        D3 = end_y - peak_y
        Total_D = D1 + D2 + D3
        
        if Total_D == 0:
            x, y = start_x, start_y
        else:
            t1 = D1 / Total_D
            t2 = (D1 + D2) / Total_D
            
            if t_eased <= t1:
                segment_t = t_eased / t1 if t1 > 0 else 0
                x = start_x
                y = start_y + (peak_y - start_y) * segment_t
            elif t_eased <= t2:
                segment_t = (t_eased - t1) / (t2 - t1) if t2 > t1 else 0
                x = start_x + (end_x - start_x) * segment_t
                y = peak_y
            else:
                segment_t = (t_eased - t2) / (1.0 - t2) if 1.0 > t2 else 0
                x = end_x
                y = peak_y + (end_y - peak_y) * segment_t
                
        img = bg_erased.copy()
        
        # Draw static stacks on the left
        for s_idx, stack in enumerate(stacks):
            for b_idx, b_name in enumerate(stack):
                bx = 79 + s_idx * 128
                by = 759 - b_idx * 64
                draw_block(img, b_name, bx, by)
                
        # Draw the moving block
        draw_block(img, blk, int(round(x)), int(round(y)))
        
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
    # Complete move
    stacks[dst].append(blk)

# Pad to exactly 150 frames
while len(frames) < 150:
    img = bg_erased.copy()
    for s_idx, stack in enumerate(stacks):
        for b_idx, b_name in enumerate(stack):
            bx = 79 + s_idx * 128
            by = 759 - b_idx * 64
            draw_block(img, b_name, bx, by)
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Save video
writer = imageio.get_writer(
    '/app/output/video.mp4', 
    fps=16, 
    codec='libx264', 
    pixelformat='yuv420p', 
    macro_block_size=None
)
for f in frames:
    writer.append_data(f)
writer.close()
print(f"Video saved with {len(frames)} frames")
