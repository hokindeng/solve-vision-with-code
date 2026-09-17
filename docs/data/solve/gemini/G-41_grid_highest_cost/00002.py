import cv2
import numpy as np
import subprocess
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    clean_bg = img.copy()
    green = np.array([80, 175, 76], dtype=np.uint8)
    clean_bg[2:254, 2:254] = green

    start_cell = img[2:254, 2:254]
    diff = cv2.absdiff(start_cell, clean_bg[2:254, 2:254])
    mask = np.any(diff > 0, axis=-1)
    
    y_coords, x_coords = np.where(mask)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    
    pacman_bgr = start_cell[min_y:max_y+1, min_x:max_x+1]
    pacman_mask = mask[min_y:max_y+1, min_x:max_x+1]
    
    base_offset_x = min_x + 2
    base_offset_y = min_y + 2
    
    path = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (3, 1), (2, 1), (1, 1), (0, 1),
        (0, 2), (0, 3), (1, 3), (1, 2),
        (2, 2), (2, 3), (3, 3)
    ]
    
    os.makedirs('/app/frames', exist_ok=True)
    frames_per_step = 6
    start_pause = 4
    total_frames = 91
    
    def get_pos_and_rot(frame_idx):
        if frame_idx < start_pause:
            return base_offset_y, base_offset_x, 'R'
            
        move_frame = frame_idx - start_pause
        
        if move_frame >= (len(path) - 1) * frames_per_step:
            r, c = path[-1]
            # The last step is from (2,3) to (3,3), so the direction is 'D'
            return r * 256 + base_offset_y, c * 256 + base_offset_x, 'D'
            
        step = move_frame // frames_per_step
        progress = (move_frame % frames_per_step) / frames_per_step
        
        r1, c1 = path[step]
        r2, c2 = path[step+1]
        
        y = (r1 + (r2 - r1) * progress) * 256 + base_offset_y
        x = (c1 + (c2 - c1) * progress) * 256 + base_offset_x
        
        if r2 > r1:
            rot = 'D'
        elif r2 < r1:
            rot = 'U'
        elif c2 > c1:
            rot = 'R'
        else:
            rot = 'L'
            
        return y, x, rot
        
    for i in range(total_frames):
        y, x, rot = get_pos_and_rot(i)
        y, x = int(round(y)), int(round(x))
        
        if rot == 'R':
            sprite = pacman_bgr
            s_mask = pacman_mask
        elif rot == 'D':
            sprite = np.rot90(pacman_bgr, -1)
            s_mask = np.rot90(pacman_mask, -1)
        elif rot == 'L':
            sprite = np.rot90(pacman_bgr, 2)
            s_mask = np.rot90(pacman_mask, 2)
        elif rot == 'U':
            sprite = np.rot90(pacman_bgr, 1)
            s_mask = np.rot90(pacman_mask, 1)
            
        frame = clean_bg.copy()
        
        h, w = sprite.shape[:2]
        if y >= 0 and y+h <= 1024 and x >= 0 and x+w <= 1024:
            bg_patch = frame[y:y+h, x:x+w]
            mask_3d = s_mask[..., None]
            frame[y:y+h, x:x+w] = np.where(mask_3d, sprite, bg_patch)
            
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)

    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)

if __name__ == '__main__':
    solve()
