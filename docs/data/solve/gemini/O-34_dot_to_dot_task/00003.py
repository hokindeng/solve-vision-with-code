import cv2
import numpy as np
import os
import subprocess
import shutil

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        print("Could not read /app/first_frame.png")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pts = [
        (256, 720), # 1
        (644, 194), # 2
        (168, 135), # 3
        (880, 438)  # 4
    ]
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    os.makedirs('/app/output', exist_ok=True)
    temp_dir = '/app/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    frames = []
    frames.append(img.copy())
    
    frames_per_segment = 54 // 3 # 18 frames per segment
    
    for f in range(1, 55):
        drawn = img.copy()
        
        seg_idx = (f - 1) // frames_per_segment
        if seg_idx > 2:
            seg_idx = 2
            
        progress = ((f - 1) % frames_per_segment + 1) / frames_per_segment
        if f == 54:
            progress = 1.0 
        
        for i in range(seg_idx):
            cv2.line(drawn, pts[i], pts[i+1], (0, 0, 255), 5, cv2.LINE_AA)
            
        start_pt = pts[seg_idx]
        end_pt = pts[seg_idx + 1]
        
        cur_x = int(start_pt[0] + progress * (end_pt[0] - start_pt[0]))
        cur_y = int(start_pt[1] + progress * (end_pt[1] - start_pt[1]))
        
        cv2.line(drawn, start_pt, (cur_x, cur_y), (0, 0, 255), 5, cv2.LINE_AA)
        
        final_frame = np.where(mask[:, :, None] == 255, img, drawn)
        frames.append(final_frame)
        
    for i, frame in enumerate(frames):
        cv2.imwrite(f'{temp_dir}/frame_{i:04d}.png', frame)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{temp_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    
    # Run ffmpeg, suppress output
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # Cleanup
    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    solve()
