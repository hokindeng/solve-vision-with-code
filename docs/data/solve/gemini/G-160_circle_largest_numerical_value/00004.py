import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None
    )
    
    cx, cy = 696, 819
    radius = 100
    thickness = 8
    color = (255, 0, 0) # RGB for red
    
    num_frames = 80
    start_arc_frame = 10
    end_arc_frame = 70
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        if i >= start_arc_frame:
            if i >= end_arc_frame:
                angle = 360
            else:
                angle = (i - start_arc_frame) / (end_arc_frame - start_arc_frame) * 360
            
            # Draw an arc
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + angle, color, thickness, lineType=cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
