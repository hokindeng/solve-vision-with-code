import cv2
import imageio
import numpy as np
import os

def generate_video():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    frames = []
    total_frames = 80
    center = (668, 682)
    radius = 100
    color = (255, 0, 0) # Red in RGB
    thickness = 5

    for i in range(total_frames):
        frame = img.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        progress = i / (total_frames - 1)
        end_angle = -90 + 360 * progress
        
        if i > 0:
            cv2.ellipse(frame_rgb, center, (radius, radius), 0, -90, end_angle, color, thickness, cv2.LINE_AA)
            
        frames.append(frame_rgb)

    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
