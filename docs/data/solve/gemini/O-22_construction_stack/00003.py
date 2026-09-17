import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    if t < 0.5: return 2 * t * t
    return -1 + (4 - 2 * t) * t

def get_path(x0, y0, x1, y1, frames):
    path = []
    peak_y = 350
    for i in range(frames):
        t = i / (frames - 1)
        tx = ease_in_out(t)
        x = x0 + (x1 - x0) * tx
        y_base = y0 + (y1 - y0) * tx
        amplitude = max(0, (y0+y1)/2 - peak_y)
        y = y_base - amplitude * np.sin(t * np.pi)
        path.append((int(x), int(y)))
    return path

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read first_frame.png")
    
    corner_row = np.full((1, 99, 3), [250, 245, 245], dtype=np.uint8)
    corner_row[0, 1] = [50, 50, 50]
    corner_row[0, 97] = [50, 50, 50]

    blocks = {}
    for b_id, (x_min, y_min) in {
        1: (79, 760),
        2: (79, 696),
        3: (79, 632),
        4: (207, 760),
        5: (207, 696)
    }.items():
        sprite_65 = img[y_min:y_min+65, x_min:x_min+99].copy()
        blocks[b_id] = np.vstack([corner_row, sprite_65])
    
    clean_bg = img.copy()
    clean_bg[400:825, 70:450] = [250, 245, 245]
    
    state = [[1, 2, 3], [4, 5], []]
    
    moves = [
        (1, 2),
        (0, 2),
        (1, 2),
        (0, 2),
        (0, 1),
        (2, 1)
    ]
    
    frames = []
    
    def render_state(moving_block=None, moving_pos=None):
        frame = clean_bg.copy()
        for s in range(3):
            x = 79 + s * 128
            for idx, b in enumerate(state[s]):
                y = 824 - 64 * (idx + 1) - 1
                frame[y:y+66, x:x+99] = blocks[b]
                
        if moving_block is not None and moving_pos is not None:
            mx, my = moving_pos
            frame[my:my+66, mx:mx+99] = blocks[moving_block]
            
        return frame

    # 8 + 6 * 17 + 10 = 120
    for _ in range(8):
        frames.append(cv2.cvtColor(render_state(), cv2.COLOR_BGR2RGB))
        
    frames_per_move = 15
    
    for src, dst in moves:
        block = state[src].pop()
        
        x_start = 79 + src * 128
        y_start = 824 - 64 * (len(state[src]) + 1) - 1
        
        x_end = 79 + dst * 128
        y_end = 824 - 64 * (len(state[dst]) + 1) - 1
        
        path = get_path(x_start, y_start, x_end, y_end, frames_per_move)
        
        for pos in path:
            frames.append(cv2.cvtColor(render_state(block, pos), cv2.COLOR_BGR2RGB))
            
        state[dst].append(block)
        
        for _ in range(2):
            frames.append(cv2.cvtColor(render_state(), cv2.COLOR_BGR2RGB))
            
    for _ in range(10):
        frames.append(cv2.cvtColor(render_state(), cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == "__main__":
    main()
