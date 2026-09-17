import cv2
import numpy as np
import imageio
import os
from PIL import Image

def main():
    # Load the first frame
    first_frame_path = '/app/first_frame.png'
    first_frame = np.array(Image.open(first_frame_path).convert('RGB'))
    
    # Create the base frame without the mask
    # We replace the mask's original position with the background color (white)
    base_frame = first_frame.copy()
    base_frame[30:281, 102:922] = [255, 255, 255]
    
    # Video parameters
    num_frames = 58
    fps = 16
    start_y = 30
    end_y = 1024
    
    frames = []
    
    for i in range(num_frames):
        # Calculate current y position for the top of the mask
        if num_frames > 1:
            current_y = int(round(start_y + i * (end_y - start_y) / (num_frames - 1)))
        else:
            current_y = start_y
            
        frame = base_frame.copy()
        
        # Calculate bounds for drawing the mask
        y1 = current_y
        y2 = current_y + 251 # Height of the mask is 251
        
        # Clip to image boundaries (1024x1024)
        draw_y1 = max(0, min(1024, y1))
        draw_y2 = max(0, min(1024, y2))
        
        if draw_y2 > draw_y1:
            frame[draw_y1:draw_y2, 102:922] = [209, 209, 209]
            
        frames.append(frame)
        
    # Write the video
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'video.mp4')
    
    # Save frames to a video using imageio
    # H.264, yuv420p required
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        macro_block_size=None, 
        codec='libx264', 
        pixelformat='yuv420p'
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    main()
