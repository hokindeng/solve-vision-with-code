import cv2
import numpy as np
import imageio.v2 as imageio
import os

def paste_block(frame, b_img, x, y):
    h, w, _ = b_img.shape
    y_int, x_int = int(round(y)), int(round(x))
    region = frame[y_int:y_int+h, x_int:x_int+w]
    mask = (b_img != [250, 245, 245]).any(axis=-1)
    region[mask] = b_img[mask]

def solve():
    img = cv2.imread('/app/first_frame.png')

    block_images = {
        'R': img[759:826, 79:178].copy(),
        'O': img[695:762, 79:178].copy(),
        'Y': img[759:826, 207:306].copy(),
        'B': img[695:762, 207:306].copy(),
        'G': img[759:826, 335:434].copy()
    }

    clean_bg = img.copy()
    clean_bg[500:826, 79:434] = [250, 245, 245]

    COL_X = {0: 79, 1: 207, 2: 335}
    LEVEL_Y = {0: 759, 1: 695, 2: 631, 3: 567}

    cols = {
        0: ['R', 'O'],
        1: ['Y', 'B'],
        2: ['G']
    }

    block_pos = {
        'R': (COL_X[0], LEVEL_Y[0]),
        'O': (COL_X[0], LEVEL_Y[1]),
        'Y': (COL_X[1], LEVEL_Y[0]),
        'B': (COL_X[1], LEVEL_Y[1]),
        'G': (COL_X[2], LEVEL_Y[0])
    }

    moves = [(0, 1), (0, 2), (1, 0), (1, 2), (1, 0), (2, 0), (2, 0), (2, 1)]

    frames = []

    # Frame 0 is exactly the original state
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Let's hold it for 4 frames
    for _ in range(3):
        frames.append(frames[0].copy())

    for move_idx, (src, dst) in enumerate(moves):
        block = cols[src].pop()
        src_level = len(cols[src])
        dst_level = len(cols[dst])
        cols[dst].append(block)
        
        start_x, start_y = COL_X[src], LEVEL_Y[src_level]
        end_x, end_y = COL_X[dst], LEVEL_Y[dst_level]
        
        # 14 frames per move:
        # 0-3: lift
        # 3-8: horiz
        # 8-11: drop
        # 11-13: rest
        for i in range(1, 15):
            if i <= 3:
                t = i / 3.0
                x = start_x
                y = start_y + t * (480 - start_y)
            elif i <= 8:
                t = (i - 3) / 5.0
                x = start_x + t * (end_x - start_x)
                y = 480
            elif i <= 11:
                t = (i - 8) / 3.0
                x = end_x
                y = 480 + t * (end_y - 480)
            else:
                x = end_x
                y = end_y
                
            frame = clean_bg.copy()
            
            static_b = [b for b in ['R', 'O', 'Y', 'B', 'G'] if b != block]
            static_b.sort(key=lambda b: block_pos[b][1], reverse=True)
            for b in static_b:
                bx, by = block_pos[b]
                paste_block(frame, block_images[b], bx, by)
                
            paste_block(frame, block_images[block], x, y)
            
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        block_pos[block] = (end_x, end_y)

    # End pause (make total ~120 frames. Currently 4 + 8*14 = 116 frames)
    # Let's add 4 more frames to reach 120 exactly
    for _ in range(4):
        frames.append(frames[-1].copy())
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')
    print("Video generated successfully.")

if __name__ == '__main__':
    solve()
