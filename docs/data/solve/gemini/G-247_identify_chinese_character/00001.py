import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # Convert BGR to RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frames = []
    num_frames = 48
    
    # Coordinates for the Chinese character "道"
    cx, cy = 665, 411
    radius = 95
    thickness = 6

    for i in range(num_frames):
        frame = img_rgb.copy()
        if i > 0:
            # calculate progression from 0 to 1
            progress = i / (num_frames - 1)
            end_angle = -90 + 360 * progress
            
            # draw arc from -90 (top) to end_angle clockwise
            # red color in RGB is (255, 0, 0)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, end_angle, (255, 0, 0), thickness, cv2.LINE_AA)
        
        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    
    # write to mp4 with H.264 and yuv420p as requested
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    print("Video generated successfully at /app/output/video.mp4")

if __name__ == "__main__":
    main()
