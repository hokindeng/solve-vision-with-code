import numpy as np
import imageio.v3 as iio
import cv2
import os

def solve():
    # Read the first frame (RGB)
    img = iio.imread('/app/first_frame.png')
    
    # Target center and radius for the Chinese character "道"
    cx, cy = 504, 173
    radius = 100
    thickness = 6
    color = (255, 0, 0) # Red in RGB
    
    num_frames = 48
    fps = 16
    
    start_frame = 5
    end_frame = 42
    
    frames = []
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i >= start_frame:
            if i >= end_frame:
                angle = 360
            else:
                # Easing could be nice, but linear is fine and safe.
                progress = (i - start_frame) / (end_frame - start_frame)
                angle = int(360 * progress)
            
            frame = np.ascontiguousarray(frame)
            
            # Start angle at -90 (top of the circle)
            cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, color, thickness, cv2.LINE_AA)
            
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    iio.imwrite('/app/output/video.mp4', frames, fps=fps, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
