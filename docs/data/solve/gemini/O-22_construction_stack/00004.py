import cv2
import numpy as np
import imageio
import math

def get_universal_mask(img):
    blue_raw = img[759:759+67, 847:847+99]
    bg = np.array([250, 245, 245])
    return ~np.all(blue_raw == bg, axis=-1)

def extract_block(img, x, y, mask):
    block = img[y:y+67, x:x+99].copy()
    rgba = cv2.cvtColor(block, cv2.COLOR_BGR2BGRA)
    rgba[~mask, 3] = 0
    return rgba

def composite(bg, fg, x, y):
    h, w = fg.shape[:2]
    bg_crop = bg[y:y+h, x:x+w]
    alpha = fg[:, :, 3:4] / 255.0
    fg_bgr = fg[:, :, :3]
    bg[y:y+h, x:x+w] = (fg_bgr * alpha + bg_crop * (1 - alpha)).astype(np.uint8)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    mask = get_universal_mask(img)
    
    blocks = {}
    blocks['B'] = extract_block(img, 847, 759, mask)
    blocks['Y'] = extract_block(img, 719, 695, mask)
    blocks['P'] = extract_block(img, 591, 695, mask)
    blocks['G'] = extract_block(img, 79, 631, mask)
    
    red_covered = extract_block(img, 591, 759, mask)
    red_covered[0:3] = blocks['B'][0:3]
    blocks['R'] = red_covered
    
    bg_clean = img.copy()
    bg_clean[631:826, 79:178] = [250, 245, 245]
    bg_clean[695:826, 207:306] = [250, 245, 245]
    
    stacks = [ ['P', 'Y', 'G'], ['R', 'B'], [] ]
    stack_x = [79, 207, 335]
    
    moves = [
        (1, 2),
        (1, 2),
        (0, 1),
        (0, 1),
        (0, 1),
        (2, 0),
        (1, 0)
    ]
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    def render_static(current_stacks):
        frame = bg_clean.copy()
        for i, stack in enumerate(current_stacks):
            for level, color in enumerate(stack):
                cx = stack_x[i]
                cy = 759 - level * 64
                composite(frame, blocks[color], cx, cy)
        return frame
    
    # Initial hold (4 frames)
    initial_frame = render_static(stacks)
    for _ in range(4):
        writer.append_data(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB))
    
    frames_per_move = 18
    
    for src, dst in moves:
        moving_color = stacks[src].pop()
        start_x = stack_x[src]
        start_y = 759 - len(stacks[src]) * 64
        
        end_x = stack_x[dst]
        end_y = 759 - len(stacks[dst]) * 64
        
        arc_height = 150 + abs(end_x - start_x) * 0.5
        
        for f in range(frames_per_move):
            t = f / float(frames_per_move - 1)
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t - arc_height * math.sin(math.pi * t)
            
            frame = render_static(stacks)
            composite(frame, blocks[moving_color], int(round(x)), int(round(y)))
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        stacks[dst].append(moving_color)
    
    # Final hold (5 frames)
    final_frame = render_static(stacks)
    for _ in range(5):
        writer.append_data(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
        
    writer.close()
    print("Video generated at /app/output/video.mp4")

if __name__ == '__main__':
    main()
