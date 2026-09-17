import cv2
import numpy as np
import os
import subprocess

def create_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    os.makedirs('/app/output', exist_ok=True)
    video_path = '/app/output/video.mp4'
    temp_path = '/app/output/temp.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 16.0, (1024, 1024))
    
    y_start = 513
    y_end = 893
    x = 497
    
    num_frames = 50
    for i in range(num_frames):
        frame = first_frame.copy()
        
        y_current = int(y_start + (y_end - y_start) * i / (num_frames - 1))
        
        if y_current > y_start:
            cv2.line(frame, (x, y_start), (x, y_current), (0, 0, 255), 4)
            
        out.write(frame)
        
    out.release()
    
    # re-encode to yuv420p as required
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', temp_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(temp_path, video_path)

if __name__ == '__main__':
    create_video()
