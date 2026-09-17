import cv2
import numpy as np
import subprocess
import os

def generate_video():
    base_img = cv2.imread('/app/first_frame.png')
    if base_img is None:
        raise FileNotFoundError("first_frame.png not found")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_path = '/tmp/temp_video.mp4'
    out = cv2.VideoWriter(temp_path, fourcc, 16.0, (1024, 1024))
    
    y_eq = 527.3333333333334
    y0_init = 294
    y1_init = 559
    y2_init = 734
    k = 0.82
    fps = 16.0
    frames = 125
    
    x_ranges = [
        (223, 313),
        (457, 547),
        (691, 781)
    ]
    
    y_start_fill = 250
    y_end_fill = 880
    
    white = [255, 255, 255]
    maroon = [75, 75, 165]
    
    for f in range(frames):
        t = f / fps
        
        y0_t = y_eq + (y0_init - y_eq) * np.exp(-k * t)
        y1_t = y_eq + (y1_init - y_eq) * np.exp(-k * t)
        y2_t = y_eq + (y2_init - y_eq) * np.exp(-k * t)
        
        y0_level = int(round(y0_t))
        y1_level = int(round(y1_t))
        y2_level = int(round(y2_t))
        
        levels = [y0_level, y1_level, y2_level]
        
        img = base_img.copy()
        
        for i in range(3):
            x_start, x_end = x_ranges[i]
            y_level = levels[i]
            
            if y_level > y_start_fill:
                img[y_start_fill:y_level, x_start:x_end] = white
                
            if y_end_fill > y_level:
                img[y_level:y_end_fill, x_start:x_end] = maroon
                
        out.write(img)
        
    out.release()
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path, 
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', out_path
    ], check=True)

if __name__ == "__main__":
    generate_video()
