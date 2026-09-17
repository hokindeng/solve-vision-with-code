import cv2
import numpy as np
import imageio
import os

def solve():
    # Input image path
    input_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the first frame
    img = cv2.imread(input_path)
    
    # Video properties
    fps = 16
    total_frames = 60
    
    # Initialize the video writer
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec='libx264', 
        pixelformat='yuv420p',
        quality=10,
        macro_block_size=None
    )
    
    # Circle parameters
    # The tangent point of the two touching circles is at (511.5, 380.0)
    # Using shift=1 for subpixel accuracy in cv2.ellipse
    center_x = int(511.5 * 2)
    center_y = int(380.0 * 2)
    radius = int(30 * 2)
    thickness = 4
    color = (0, 0, 0) # Black
    
    for i in range(total_frames):
        frame = img.copy()
        
        # Calculate how much of the circle to draw based on current frame
        progress = i / (total_frames - 1)
        end_angle = int(progress * 360)
        
        if end_angle > 0:
            cv2.ellipse(
                frame, 
                (center_x, center_y), 
                (radius, radius), 
                0, 
                0, 
                end_angle, 
                color, 
                thickness, 
                cv2.LINE_AA,
                shift=1
            )
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    solve()
