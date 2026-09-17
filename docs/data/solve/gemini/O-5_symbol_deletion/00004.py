import cv2
import numpy as np
import imageio
import os

def solve():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bg_color = img_rgb[0, 0]
    
    # Red border is purely red (R>200, G<50, B<50)
    red_mask = (img_rgb[:, :, 0] > 200) & (img_rgb[:, :, 1] < 50) & (img_rgb[:, :, 2] < 50)
    y, x = np.where(red_mask)
    
    if len(x) == 0:
        print("No red border found!")
        return
        
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    num_frames = 46
    frames = []
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        # alpha goes from 1.0 to 0.0
        alpha = 1.0 - (i / (num_frames - 1))
        
        # Extract the region containing the symbol and red border
        region = frame[y_min:y_max+1, x_min:x_max+1]
        
        # Blend it with the background color
        blended_region = region.astype(float) * alpha + bg_color.astype(float) * (1.0 - alpha)
        frame[y_min:y_max+1, x_min:x_max+1] = np.round(blended_region).astype(np.uint8)
        
        frames.append(frame)
        
    # Write video with the exact required parameters
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')
    print("Video generated successfully.")

if __name__ == '__main__':
    solve()
