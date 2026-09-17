import cv2
import numpy as np
import imageio
import os

def make_video():
    first_frame_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise FileNotFoundError(f"Could not read {first_frame_path}")
        
    # Bathroom coordinates
    x1, y1 = 60, 702
    x2, y2 = 376, 964
    
    thickness = 8
    
    w = x2 - x1
    h = y2 - y1
    total_length = 2 * (w + h)
    
    frames = []
    num_frames = 28
    
    for i in range(num_frames):
        frame = first_frame.copy()
        
        progress = i / (num_frames - 1)
        current_length = int(total_length * progress)
        
        if current_length > 0:
            top_len = min(current_length, w)
            cv2.line(frame, (x1, y1), (x1 + top_len, y1), (0, 255, 0), thickness)
            current_length -= top_len
            
        if current_length > 0:
            right_len = min(current_length, h)
            cv2.line(frame, (x2, y1), (x2, y1 + right_len), (0, 255, 0), thickness)
            current_length -= right_len
            
        if current_length > 0:
            bottom_len = min(current_length, w)
            cv2.line(frame, (x2, y2), (x2 - bottom_len, y2), (0, 255, 0), thickness)
            current_length -= bottom_len
            
        if current_length > 0:
            left_len = min(current_length, h)
            cv2.line(frame, (x1, y2), (x1, y2 - left_len), (0, 255, 0), thickness)
            current_length -= left_len
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    make_video()
