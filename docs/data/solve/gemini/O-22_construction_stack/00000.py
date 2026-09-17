import cv2
import numpy as np
import imageio
import os

def ease(t):
    if t < 0: return 0.0
    if t > 1: return 1.0
    return t * t * (3 - 2 * t)

def get_pos(src_x, src_y, dst_x, dst_y, t):
    y_lift = 550
    if t < 0.2:
        p = t / 0.2
        p = ease(p)
        return src_x, src_y + (y_lift - src_y) * p
    elif t < 0.8:
        p = (t - 0.2) / 0.6
        p = ease(p)
        return src_x + (dst_x - src_x) * p, y_lift
    else:
        p = (t - 0.8) / 0.2
        p = ease(p)
        return dst_x, y_lift + (dst_y - y_lift) * p

def paste_block(frame, block_img, x, y):
    x = int(round(x))
    y = int(round(y))
    h, w = block_img.shape[:2]
    # mask for non-(250, 245, 245) pixels
    mask = (block_img[:, :, 0] != 250) | (block_img[:, :, 1] != 245) | (block_img[:, :, 2] != 245)
    
    y1, y2 = max(0, y), min(frame.shape[0], y + h)
    x1, x2 = max(0, x), min(frame.shape[1], x + w)
    
    by1, by2 = y1 - y, y2 - y
    bx1, bx2 = x1 - x, x2 - x
    
    if y1 >= y2 or x1 >= x2:
        return
        
    sub_mask = mask[by1:by2, bx1:bx2]
    frame[y1:y2, x1:x2][sub_mask] = block_img[by1:by2, bx1:bx2][sub_mask]

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not load /app/first_frame.png")
        
    bg = img.copy()
    bg[759:826, 79:178] = (250, 245, 245)
    bg[695:826, 335:434] = (250, 245, 245)
    
    orange_clean = img[759:759+67, 79:79+99].copy()
    
    green_clean = img[695:695+67, 335:335+99].copy()
    green_clean[66, :] = orange_clean[66, :]
    
    purple_clean = img[759:759+67, 335:335+99].copy()
    purple_clean[0, :] = orange_clean[0, :]
    
    blocks = {
        'O': orange_clean,
        'G': green_clean,
        'P': purple_clean
    }
    
    bases_x = {
        1: 79,
        2: 207,
        3: 335
    }
    
    state = {
        1: ['O'],
        2: [],
        3: ['P', 'G']
    }
    
    moves = [
        (3, 1), # G to 1
        (3, 2), # P to 2
        (1, 2), # G to 2
        (1, 2)  # O to 2
    ]
    
    frames = []
    
    # 5 frames pause
    for _ in range(5):
        frame = bg.copy()
        for b_id, stack in state.items():
            for i, block_name in enumerate(stack):
                x = bases_x[b_id]
                y = 759 - i * 64
                paste_block(frame, blocks[block_name], x, y)
        frames.append(frame)
        
    # Animate
    for move in moves:
        src, dst = move
        moving_block_name = state[src].pop()
        
        src_x = bases_x[src]
        src_y = 759 - len(state[src]) * 64
        
        dst_x = bases_x[dst]
        dst_y = 759 - len(state[dst]) * 64
        
        for f in range(20):
            t = f / 19.0
            frame = bg.copy()
            
            for b_id, stack in state.items():
                for i, block_name in enumerate(stack):
                    x = bases_x[b_id]
                    y = 759 - i * 64
                    paste_block(frame, blocks[block_name], x, y)
            
            cur_x, cur_y = get_pos(src_x, src_y, dst_x, dst_y, t)
            paste_block(frame, blocks[moving_block_name], cur_x, cur_y)
            
            frames.append(frame)
            
        state[dst].append(moving_block_name)
        
    # 5 frames pause
    for _ in range(5):
        frame = bg.copy()
        for b_id, stack in state.items():
            for i, block_name in enumerate(stack):
                x = bases_x[b_id]
                y = 759 - i * 64
                paste_block(frame, blocks[block_name], x, y)
        frames.append(frame)
        
    # Write video
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    for f in frames_rgb:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    main()
