import cv2
import numpy as np
import imageio
import os

def main():
    img_path = '/app/first_frame.png'
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'video.mp4')
    
    # Read the first frame
    frame = cv2.imread(img_path)
    
    # Circle parameters
    cx, cy = 717, 248
    radius = 110
    color = (0, 0, 255) # Red in BGR
    thickness = 8
    
    num_frames = 80
    fps = 16
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    start_frame = 10
    end_frame = 70
    
    for i in range(num_frames):
        out_frame = frame.copy()
        
        if i >= start_frame:
            if i >= end_frame:
                progress = 1.0
            else:
                progress = (i - start_frame) / (end_frame - start_frame)
            
            end_angle = 360 * progress
            
            # Draw the arc
            # cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType)
            # axes is (major_axis_length, minor_axis_length) which is (radius, radius) for a circle
            # -90 angle starts it at the top
            cv2.ellipse(out_frame, (cx, cy), (radius, radius), -90, 0, end_angle, color, thickness, cv2.LINE_AA)
            
        # imageio expects RGB
        out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(out_frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
