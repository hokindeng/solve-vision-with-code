import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Find green object mask
    # The green border defines the object
    green_mask = cv2.inRange(img, np.array([0, 150, 0]), np.array([50, 255, 50]))
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    
    # Create mask covering the whole object
    obj_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(obj_mask, [c], -1, 255, -1)
    
    # 2. Create background by removing the green object
    bg_image = img_rgb.copy()
    bg_image[obj_mask > 0] = [255, 255, 255]
    
    # 3. Determine movement parameters
    # The red star object's center is at x=136
    # The green object's center is at x=308
    total_delta_x = -172
    n_frames = 60
    
    # Pre-calculate region of interest for slicing to avoid full-frame operations
    y_coords, x_coords = np.where(obj_mask > 0)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    mask_roi = obj_mask[y_min:y_max+1, x_min:x_max+1] > 0
    src_roi = img_rgb[y_min:y_max+1, x_min:x_max+1]
    
    os.makedirs('/app/output', exist_ok=True)
    
    # Initialize video writer
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        macro_block_size=None, 
        pixelformat='yuv420p'
    )
    
    # Generate and write frames
    for i in range(n_frames):
        # Linear interpolation
        delta_x = int(round(total_delta_x * i / (n_frames - 1)))
        
        frame = bg_image.copy()
        
        dst_x_min = x_min + delta_x
        dst_x_max = x_max + delta_x
        
        dst_roi = frame[y_min:y_max+1, dst_x_min:dst_x_max+1]
        dst_roi[mask_roi] = src_roi[mask_roi]
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
