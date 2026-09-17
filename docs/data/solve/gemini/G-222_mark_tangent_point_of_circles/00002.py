import cv2
import numpy as np
import subprocess
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    height, width, _ = img.shape
    
    # Intersection center mathematically found
    center = (453, 361)
    radius = 30
    color = (0, 0, 0)
    thickness = 4
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = '/app/output/temp_video.mp4'
    final_video = '/app/output/video.mp4'
    
    os.makedirs('/app/output', exist_ok=True)
    
    out = cv2.VideoWriter(temp_video, fourcc, 16.0, (width, height))
    
    total_frames = 60
    
    for i in range(total_frames):
        frame = img.copy()
        
        angle = int(360 * i / (total_frames - 1))
        
        if angle > 0:
            cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, color, thickness, cv2.LINE_AA)
            
        out.write(frame)
        
    out.release()
    
    # convert to h264 yuv420p with a good quality setting
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_video, 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', final_video
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(temp_video)

if __name__ == '__main__':
    create_video()
