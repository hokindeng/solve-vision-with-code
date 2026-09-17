import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    green_color_rgb = np.array([70, 140, 70])
    mask = cv2.inRange(img_rgb, green_color_rgb, green_color_rgb)
    y_coords, x_coords = np.where(mask > 0)
    
    base_img = img_rgb.copy()
    base_img[mask > 0] = [255, 255, 255]
    
    # Target translation based on object centers
    # Left object center: (276.0, 511.5)
    # Right object center: (797.0, 425.5)
    target_dx = 797.0 - 276.0
    target_dy = 425.5 - 511.5
    
    num_frames = 25
    frames = []
    
    for i in range(num_frames):
        progress = i / (num_frames - 1)
        # Apply smoothstep (ease-in-out) for a nicer animation
        smooth_progress = progress * progress * (3 - 2 * progress)
        
        dx = int(round(target_dx * smooth_progress))
        dy = int(round(target_dy * smooth_progress))
        
        frame = base_img.copy()
        
        new_x = x_coords + dx
        new_y = y_coords + dy
        
        valid = (new_x >= 0) & (new_x < frame.shape[1]) & (new_y >= 0) & (new_y < frame.shape[0])
        frame[new_y[valid], new_x[valid]] = green_color_rgb
        
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
