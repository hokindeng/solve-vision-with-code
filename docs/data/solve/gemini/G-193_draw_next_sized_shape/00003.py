import cv2
import numpy as np
import os
import subprocess
import shutil

def main():
    img = cv2.imread('/app/first_frame.png')
    
    output_dir = '/app/frames'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Target square
    x_start = 857
    y_start = 485
    w = 55
    h = 55
    color = [216, 78, 29] # BGR
    
    total_frames = 60
    
    for i in range(total_frames):
        frame = img.copy()
        
        # Calculate how many rows to draw
        # Start drawing at frame 1, finish at frame 55
        rows_to_draw = 0
        if i > 0:
            rows_to_draw = min(i, h)
            
        if rows_to_draw > 0:
            frame[y_start : y_start + rows_to_draw, x_start : x_start + w] = color
            
        cv2.imwrite(f'{output_dir}/frame_{i:04d}.png', frame)
        
    # generate video
    os.makedirs('/app/output', exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{output_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
if __name__ == '__main__':
    main()
