import cv2
import numpy as np
import imageio
import os

def create_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    start_y, start_x = 700, 235
    cell = first_frame[start_y:start_y+90, start_x:start_x+90]
    blue = np.array([255, 100, 0])
    
    # Extract agent mask and pixels
    diff = np.sum(np.abs(cell.astype(int) - blue), axis=-1)
    agent_mask = diff > 10
    agent_pixels = cell.copy()
    
    # Create clean background
    clean_frame = first_frame.copy()
    clean_frame[start_y:start_y+90, start_x:start_x+90] = blue
    
    frames = []
    total_frames = 22
    end_x = 514
    
    for i in range(total_frames):
        frame = clean_frame.copy()
        
        # Calculate current position using linear interpolation
        curr_x = int(round(start_x + i * (end_x - start_x) / (total_frames - 1)))
        curr_y = start_y
        
        # Place agent over the current background
        roi = frame[curr_y:curr_y+90, curr_x:curr_x+90]
        roi[agent_mask] = agent_pixels[agent_mask]
        frame[curr_y:curr_y+90, curr_x:curr_x+90] = roi
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    create_video()
