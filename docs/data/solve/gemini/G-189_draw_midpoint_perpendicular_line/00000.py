import cv2
import numpy as np
import imageio
import os

def generate_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    frames = []
    
    total_frames = 50
    start_y = 334
    end_y = 612
    x_start = 805
    x_end = 809 # 4 pixels wide: 805, 806, 807, 808. Midpoint 806.5.
    
    # The point is at x=807. Since a 4-pixel width can't perfectly center on an integer pixel,
    # 805 to 808 covers the point. Or we could use 5 pixels (805 to 809, midpoint 807).
    # Let's use 5 pixels to be perfectly symmetric around 807, just like cv2.line(..., 4) does.
    x_start = 805
    x_end = 810
    
    for i in range(total_frames):
        frame = first_frame.copy()
        
        if i == 0:
            current_y = start_y - 1
        else:
            progress = i / (total_frames - 1)
            current_y = start_y + int((end_y - start_y) * progress)
            
        if current_y >= start_y:
            # We want to draw a red line from start_y to current_y
            # Only overwrite white pixels to keep other elements (like the point) unchanged
            
            # Slice the region
            region = frame[start_y : current_y + 1, x_start : x_end]
            
            # Identify white pixels in the region
            white_mask = (region == [255, 255, 255]).all(axis=-1)
            
            # Color them red [0, 0, 255]
            region[white_mask] = [0, 0, 255]
                    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
