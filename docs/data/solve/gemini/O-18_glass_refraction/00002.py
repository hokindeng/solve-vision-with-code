import cv2
import numpy as np
import math
import os
import subprocess

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Erase angle annotation
    # Erase arc (left side)
    for i in range(465, 485):
        for j in range(480, 515):
            if img[i, j, 0] < 200: # not blue (B < 200)
                img[i, j] = [255, 255, 255]
                
    # Erase text (right side)
    for i in range(460, 510):
        for j in range(540, 650):
            if img[i, j, 0] < 250: # text is black/gray
                img[i, j] = [255, 255, 255]
                
    # 2. Precalculate the refracted ray
    theta_i = math.radians(30.0)
    n_air = 1.00
    n_glass = 1.630
    sin_theta_r = (n_air * math.sin(theta_i)) / n_glass
    theta_r = math.asin(sin_theta_r)
    tan_theta_r = math.tan(theta_r)
    
    # End point
    # dy goes from 0 to 512 (since y goes from 512 to 1024)
    max_dy = 512
    max_dx = max_dy * tan_theta_r
    
    # The normal line segment to restore
    normal_line = img[482:543, 512:513].copy()
    
    # Generate frames
    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 70
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Draw refracted ray
        frac = i / (num_frames - 1)
        if frac > 0:
            dy = max_dy * frac
            dx = dy * tan_theta_r
            x_t = int(round(512 + dx))
            y_t = int(round(512 + dy))
            cv2.line(frame, (512, 512), (x_t, y_t), (0, 0, 255), 4, cv2.LINE_AA)
            
            # Restore the normal line segment over the ray
            frame[482:543, 512:513] = normal_line
            
        cv2.imwrite(f"{frames_dir}/frame_{i:04d}.png", frame)
        
    # Create video
    cmd = [
        "ffmpeg", "-y", "-framerate", "16", "-i", f"{frames_dir}/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "/app/output/video.mp4"
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    solve()
