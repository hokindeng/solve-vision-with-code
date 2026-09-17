import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Pure blue is RGB (0, 0, 255).
    mask = np.all(img_rgb == [0, 0, 255], axis=-1)
    
    os.makedirs('/app/output', exist_ok=True)
    
    num_frames = 96
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        # alpha from 1 to 0
        alpha = 1.0 - (i / (num_frames - 1))
        
        # Current color
        current_color = np.array([0, 0, 255]) * alpha + np.array([255, 255, 255]) * (1 - alpha)
        current_color = current_color.astype(np.uint8)
        
        frame[mask] = current_color
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
