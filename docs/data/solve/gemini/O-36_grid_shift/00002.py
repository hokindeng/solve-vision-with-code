import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    bg = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    gray_lines = [0, 113, 227, 341, 455, 568, 682, 796, 910]
    for loc in gray_lines:
        bg[loc, :] = [51, 51, 51]
        bg[:, loc] = [51, 51, 51]
        
    black_mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    contours, _ = cv2.findContours(black_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        block_img = img[y:y+h, x:x+w].copy()
        
        c_y = 0
        for i, gl in enumerate(gray_lines):
            if gl <= y:
                c_y = i
        
        target_y = y + (gray_lines[c_y+1] - gray_lines[c_y]) if c_y + 1 < len(gray_lines) else y + (1024 - gray_lines[c_y])
            
        blocks.append({
            'x': x,
            'y_start': y,
            'y_end': target_y,
            'w': w,
            'h': h,
            'img': block_img
        })
        
    os.makedirs('/app/output', exist_ok=True)
    frames = []
    num_frames = 35
    
    for f in range(num_frames):
        frame = bg.copy()
        for b in blocks:
            current_y = int(round(b['y_start'] + (b['y_end'] - b['y_start']) * (f / (num_frames - 1))))
            frame[current_y:current_y+b['h'], b['x']:b['x']+b['w']] = b['img']
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, format='FFMPEG', codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
