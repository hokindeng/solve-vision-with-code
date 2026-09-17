import cv2
import numpy as np
import os
import subprocess

def main():
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise ValueError("Cannot read /app/first_frame.png")
    
    # 1. Create clean background
    # Erase the question mark which is roughly at x=756, y=746, w=27, h=30
    mask = (first_frame[:, :, 0] < 150) & (first_frame[:, :, 1] < 150) & (first_frame[:, :, 2] < 150) & (first_frame[:, :, 0] > 100)
    area_mask = np.zeros_like(mask)
    area_mask[730:790, 730:790] = True
    q_mask = mask & area_mask
    
    clean_bg = first_frame.copy()
    clean_bg[q_mask] = [255, 255, 255]
    
    # 2. Extract C
    C = first_frame[693:693+151, 180:180+151]
    alpha = ((C[:, :, 0] < 255) | (C[:, :, 1] < 255) | (C[:, :, 2] < 255)).astype(np.uint8) * 255
    C_rgba = cv2.cvtColor(C, cv2.COLOR_BGR2BGRA)
    C_rgba[:, :, 3] = alpha
    C_rgba[alpha == 0] = [0, 0, 0, 0]
    
    # 3. Animation settings
    num_frames = 60
    fps = 16
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'video.mp4')
    
    # Create a VideoWriter or write frames and use ffmpeg
    # Using ffmpeg directly via subprocess for best compatibility and exact params
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Target scale
    target_scale = 131.0 / 151.0
    
    # Animation
    for t in range(num_frames):
        progress = t / float(num_frames - 1) # 0.0 to 1.0
        
        # Background fade (fades out the ?)
        frame = clean_bg.astype(np.float32)
        frame += (first_frame.astype(np.float32) - clean_bg.astype(np.float32)) * (1.0 - progress)
        frame = frame.astype(np.uint8)
        
        # Current scale and position
        current_scale = 1.0 + (target_scale - 1.0) * progress
        cx = 255.5 + 514.0 * progress
        cy = 768.5
        
        # Resize C
        # Note: if scale == 1.0, resize doesn't change anything. 
        # But we must be careful with integer sizes
        new_w = int(round(151 * current_scale))
        new_h = int(round(151 * current_scale))
        
        C_res = cv2.resize(C_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Top-left corner for placement
        x1 = int(round(cx - new_w / 2.0))
        y1 = int(round(cy - new_h / 2.0))
        x2 = x1 + new_w
        y2 = y1 + new_h
        
        # Composite C_res onto frame
        a = C_res[:, :, 3] / 255.0
        for c in range(3):
            frame[y1:y2, x1:x2, c] = C_res[:, :, c] * a + frame[y1:y2, x1:x2, c] * (1 - a)
            
        cv2.imwrite(os.path.join(frames_dir, f"frame_{t:04d}.png"), frame)
        
    # Compile to mp4
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-s', '1024x1024',
        out_path
    ]
    subprocess.run(cmd, check=True)
    
if __name__ == '__main__':
    main()
