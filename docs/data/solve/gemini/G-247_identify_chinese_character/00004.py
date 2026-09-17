import cv2
import numpy as np
import imageio
import os

def create_video():
    input_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    base_img = cv2.imread(input_path)
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    
    num_frames = 48
    fps = 16
    
    center = (458, 364)
    radius = 95
    color = (255, 0, 0) # Red in RGB
    thickness = 8
    
    # ensure output is yuv420p
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None
    )
    
    for i in range(num_frames):
        # Progress from 0 to 1
        progress = i / (num_frames - 1)
        angle = progress * 360
        
        frame = base_img.copy()
        
        if angle > 0:
            cv2.ellipse(
                frame, 
                center, 
                (radius, radius), 
                -90, # Starts drawing from top
                0, 
                angle, 
                color, 
                thickness,
                cv2.LINE_AA
            )
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    create_video()
