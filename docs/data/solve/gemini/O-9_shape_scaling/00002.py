import cv2
import numpy as np
import os
import subprocess

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Extract C
    C_patch = img[693:844, 203:308].copy()
    
    # Extract ?
    q_x1, q_x2 = 750, 790
    q_y1, q_y2 = 740, 790
    Q_patch = img[q_y1:q_y2, q_x1:q_x2].copy()
    
    # Base image with ? erased (filled with white)
    base_img = img.copy()
    base_img[q_y1:q_y2, q_x1:q_x2] = 255
    
    # Target center for the new scaled C
    cx = 769.5
    cy = 768.5
    
    # Scale factor from A (width 151) to B (width 131) is 131/151.
    # Applying this to C (width 105, height 151):
    # target_w = 105 * 131 / 151 = 91.09 -> 91
    # target_h = 151 * 131 / 151 = 131
    target_w = 91
    target_h = 131
    
    frames = []
    num_frames = 60
    
    for i in range(num_frames):
        frame = base_img.copy()
        
        t = i / (num_frames - 1)
        
        # Fade out ? in the first 15 frames
        if i < 15:
            alpha_q = 1.0 - (i / 15.0)
            blended_q = cv2.addWeighted(Q_patch, alpha_q, np.full_like(Q_patch, 255), 1 - alpha_q, 0)
            frame[q_y1:q_y2, q_x1:q_x2] = blended_q
            
        # Scale up C (grow from size 0 to target_size)
        # Using ease-out quadratic for smoother growth
        ease_t = 1.0 - (1.0 - t) ** 2
        
        current_w = max(1, int(target_w * ease_t))
        current_h = max(1, int(target_h * ease_t))
        
        if current_w > 1 and current_h > 1:
            C_scaled = cv2.resize(C_patch, (current_w, current_h), interpolation=cv2.INTER_NEAREST)
            mask = (C_scaled != 255).any(axis=-1)
            
            start_x = int(cx - current_w / 2)
            start_y = int(cy - current_h / 2)
            
            for y in range(current_h):
                for x in range(current_w):
                    if mask[y, x]:
                        fy = start_y + y
                        fx = start_x + x
                        if 0 <= fy < 1024 and 0 <= fx < 1024:
                            frame[fy, fx] = C_scaled[y, x]
                            
        frames.append(frame)
        
    # Force the exact first frame
    frames[0] = img.copy()
    
    os.makedirs('/app/output', exist_ok=True)
    temp_video = '/app/output/temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, 16.0, (1024, 1024))
    for f in frames:
        out.write(f)
    out.release()
    
    final_video = '/app/output/video.mp4'
    subprocess.run(['ffmpeg', '-y', '-i', temp_video, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', final_video], check=True)
    os.remove(temp_video)

if __name__ == '__main__':
    create_video()
