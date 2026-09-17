import cv2
import numpy as np
import imageio
import os

def solve():
    # Ensure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    # cv2 reads in BGR, we need RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract the source green circle (from x=34)
    # y=434:591, x=34:191 (157x157)
    src_roi = img_rgb[434:434+157, 34:34+157].copy()
    
    # Extract the placeholder ROI (from x=834)
    # y=434:591, x=834:991 (157x157)
    placeholder_roi = img_rgb[434:434+157, 834:834+157].copy()
    
    num_frames = 64
    fps = 16
    
    # Setup video writer
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        # Calculate wipe width
        # at i=0, w=0 (all placeholder)
        # at i=num_frames-1, w=157 (all src)
        w = int(round((i / (num_frames - 1)) * 157))
        
        # Create the current ROI based on wipe
        current_roi = placeholder_roi.copy()
        current_roi[:, :w] = src_roi[:, :w]
        
        # Place it back in the frame
        frame[434:434+157, 834:834+157] = current_roi
        
        # Write frame
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
