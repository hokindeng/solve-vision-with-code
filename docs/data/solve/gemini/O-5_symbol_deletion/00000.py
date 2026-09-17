import cv2
import numpy as np
import imageio
import os

def create_video():
    os.makedirs('/app/output', exist_ok=True)
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    frames = 46
    fps = 16
    
    # Red color in RGB is [255, 0, 0]
    red_mask = (first_frame[:, :, 0] == 255) & (first_frame[:, :, 1] == 0) & (first_frame[:, :, 2] == 0)
    y, x = np.where(red_mask)
    if len(y) == 0:
        writer = imageio.get_writer('/app/output/video.mp4', fps=fps, macro_block_size=1, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
        for _ in range(frames):
            writer.append_data(first_frame)
        writer.close()
        return
        
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, macro_block_size=1, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
    
    bg_color = np.array([255, 255, 255], dtype=float)
    
    for t in range(frames):
        frame = first_frame.copy()
        
        progress = t / (frames - 1)
        alpha = max(0.0, 1.0 - progress)
        
        if alpha < 1.0:
            region = first_frame[y_min:y_max+1, x_min:x_max+1].astype(float)
            blended = region * alpha + bg_color * (1.0 - alpha)
            frame[y_min:y_max+1, x_min:x_max+1] = blended.astype(np.uint8)
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    create_video()
