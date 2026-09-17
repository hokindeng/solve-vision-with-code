import numpy as np
from PIL import Image
import imageio
import os

def create_video():
    first_frame_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = Image.open(first_frame_path).convert('RGB')
    arr = np.array(img)
    
    # Identify the green box mask
    box_color = [70, 140, 70]
    box_mask = (arr[:, :, 0] == box_color[0]) & (arr[:, :, 1] == box_color[1]) & (arr[:, :, 2] == box_color[2])
    
    # Create background image: image without the green box, replaced with white
    bg_arr = arr.copy()
    bg_arr[box_mask] = [255, 255, 255]
    
    # Total frames = 25
    num_frames = 25
    fps = 16
    
    # Target shift
    dx_total = 569
    dy_total = 1
    
    frames = []
    
    # Find original coordinates of the green box pixels
    y_box, x_box = np.where(box_mask)
    
    for i in range(num_frames):
        # Calculate current shift
        if i == num_frames - 1:
            dx = dx_total
            dy = dy_total
        else:
            dx = int(round(dx_total * i / (num_frames - 1)))
            dy = int(round(dy_total * i / (num_frames - 1)))
            
        # Create frame
        frame_arr = bg_arr.copy()
        
        # Calculate new coordinates
        new_x = x_box + dx
        new_y = y_box + dy
        
        # Only draw pixels that are within bounds
        valid = (new_x >= 0) & (new_x < frame_arr.shape[1]) & (new_y >= 0) & (new_y < frame_arr.shape[0])
        
        frame_arr[new_y[valid], new_x[valid]] = box_color
        
        frames.append(frame_arr)
        
    # Save to video
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    create_video()
