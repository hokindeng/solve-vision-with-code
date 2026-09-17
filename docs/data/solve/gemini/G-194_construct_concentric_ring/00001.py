import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import subprocess

def main():
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    temp_path = '/app/output/temp.mp4'
    
    num_frames = 40
    fps = 16
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (1024, 1024))
    
    c1_start = np.array([394.0, 307.0])
    c2_start = np.array([847.0, 716.0])
    c_end = np.array([512.0, 512.0])
    
    r1 = 380
    r2 = 163
    w = 8
    
    for t in range(num_frames):
        alpha = t / (num_frames - 1)
        
        c1_curr = c1_start * (1 - alpha) + c_end * alpha
        c2_curr = c2_start * (1 - alpha) + c_end * alpha
        
        cx1 = int(round(c1_curr[0]))
        cy1 = int(round(c1_curr[1]))
        
        cx2 = int(round(c2_curr[0]))
        cy2 = int(round(c2_curr[1]))
        
        frame = Image.new('RGB', (1024, 1024), (255, 255, 255))
        draw = ImageDraw.Draw(frame)
        
        draw.ellipse([cx1-r1, cy1-r1, cx1+r1, cy1+r1], outline=(210, 245, 60), width=w)
        draw.ellipse([cx2-r2, cy2-r2, cx2+r2, cy2+r2], outline=(240, 50, 230), width=w)
        
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # In frame 0, we can verify that this matches first_frame.png, 
        # but here we just write it directly.
        out.write(frame_bgr)
        
    out.release()
    
    # Convert to H.264, yuv420p using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path, 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
        out_path
    ], check=True)
    
    os.remove(temp_path)

if __name__ == '__main__':
    main()
