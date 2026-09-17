import cv2
import numpy as np
import os
import subprocess

def create_video():
    # Read the initial image
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
    
    # Parameters for the correct option (Option 3, 4th from left)
    # The bounding box of the option is at x=751, y=752 with w=211, h=205
    box_x, box_y = 751, 752
    box_w, box_h = 211, 205
    
    # Calculate the center of the option box
    cx = box_x + box_w // 2
    cy = box_y + box_h // 2
    
    # Circle parameters
    radius = 95
    thickness = 6
    color = (0, 0, 255) # Red in BGR
    
    num_frames = 60
    fps = 16.0
    
    # Make sure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    temp_path = '/app/temp.mp4'
    out_path = '/app/output/video.mp4'
    
    # Use cv2.VideoWriter for temporary uncompressed/simple mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    
    for i in range(num_frames):
        frame = first_frame.copy()
        
        if i > 0:
            # Progress goes from 0 to 1 over the frames (excluding first frame which has 0 progress)
            progress = i / (num_frames - 1)
            end_angle = int(round(progress * 360))
            
            if end_angle > 0:
                # cv2.ellipse args: img, center, axes, angle, startAngle, endAngle, color, thickness, lineType
                cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, end_angle, color, thickness, cv2.LINE_AA)
                
        out.write(frame)
        
    out.release()
    
    # Convert to standard H.264 mp4 format with required properties
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path, 
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        '-r', str(int(fps)),
        out_path
    ], check=True)
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == '__main__':
    create_video()
