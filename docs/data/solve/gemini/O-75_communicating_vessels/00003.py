import cv2
import numpy as np
import os
import subprocess

def create_video():
    first_frame_path = '/app/first_frame.png'
    output_dir = '/app/output'
    output_path = os.path.join(output_dir, 'video.mp4')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_orig = cv2.imread(first_frame_path)
    
    tubes = [
        {"x1": 106, "x2": 195, "y0": 644},
        {"x1": 340, "x2": 429, "y0": 659},
        {"x1": 574, "x2": 663, "y0": 324},
        {"x1": 808, "x2": 897, "y0": 759}
    ]
    
    y_avg = 596.5
    k = 1.87
    fps = 16
    total_frames = 58
    
    # We will write frames to a temporary directory and then use ffmpeg
    temp_dir = '/app/temp_frames'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    for f in range(total_frames):
        t = f / fps
        img = img_orig.copy()
        
        for tube in tubes:
            x1, x2 = tube["x1"], tube["x2"]
            y0 = tube["y0"]
            
            # Calculate current y level
            y_level_f = y_avg + (y0 - y_avg) * np.exp(-k * t)
            y_level = int(round(y_level_f))
            
            # Fill white above liquid
            # Start clearing from y=300 to be safe (max initial liquid was 324)
            img[300:y_level, x1:x2+1] = [255, 255, 255]
            
            # Fill honey below liquid down to 880
            img[y_level:880, x1:x2+1] = [75, 191, 255]
            
        cv2.imwrite(os.path.join(temp_dir, f'frame_{f:04d}.png'), img)
        
    # generate video
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)

if __name__ == '__main__':
    create_video()
