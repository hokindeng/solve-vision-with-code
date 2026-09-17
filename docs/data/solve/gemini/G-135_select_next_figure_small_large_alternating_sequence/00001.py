import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Target center and radius
    # Box 0: (61, 752, 211, 205) -> Center (166, 854)
    center = (166, 854)
    radius = 95
    color = (0, 0, 255) # Red in BGR
    thickness = 8
    
    frames = []
    num_frames = 60
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            # Calculate how much of the circle to draw
            # Start at -90 degrees (top of the circle) and draw clockwise
            end_angle = -90 + (360 * i / (num_frames - 1))
            
            # We need to draw an ellipse arc
            # cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType)
            cv2.ellipse(frame, center, (radius, radius), 0, -90, end_angle, color, thickness, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    os.makedirs('/app/output', exist_ok=True)
    # Write video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
    
if __name__ == '__main__':
    main()
