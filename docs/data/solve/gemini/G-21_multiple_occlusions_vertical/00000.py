import numpy as np
from PIL import Image
import imageio
import os

def generate_video():
    # Load first frame
    img = np.array(Image.open('/app/first_frame.png').convert('RGB'))

    # The mask is at x=102..921, y=30..280 (inclusive)
    # We know the background is white [255, 255, 255]
    clean_bg = img.copy()
    clean_bg[30:281, 102:922] = [255, 255, 255]

    mask_color = [209, 209, 209]
    mask_h = 281 - 30
    mask_w = 922 - 102
    
    x_start = 102
    x_end = 922

    num_frames = 58
    fps = 16
    
    start_y = 30
    end_y = 1024

    frames = []
    for t in range(num_frames):
        frame = clean_bg.copy()
        
        # Calculate mask y position
        y = int(round(start_y + t * (end_y - start_y) / (num_frames - 1)))
        
        y_start = y
        y_end = y + mask_h
        
        # Clip to image boundaries
        draw_y_start = max(0, y_start)
        draw_y_end = min(1024, y_end)
        
        if draw_y_start < draw_y_end:
            frame[draw_y_start:draw_y_end, x_start:x_end] = mask_color
            
        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=fps, macro_block_size=1, codec='libx264', pixelformat='yuv420p', quality=10)

if __name__ == '__main__':
    generate_video()
