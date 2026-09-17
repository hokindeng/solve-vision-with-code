import cv2
import numpy as np
import scipy.ndimage as ndimage
import imageio
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # 5th shape (placeholder)
    # The bounding box of the 5th shape is 832:832+161, 432:432+161
    roi5 = img[432:432+161, 832:832+161]
    gray_mask = np.all(roi5 == [100, 100, 100], axis=-1)
    
    # Fill the outline to get the interior
    mask5_filled = ndimage.binary_fill_holes(gray_mask)
    mask5_interior = mask5_filled ^ gray_mask
    
    # Find bounding box of interior
    y_coords, _ = np.where(mask5_interior)
    min_y = y_coords.min()
    max_y = y_coords.max()
    
    # Target color is Blue: [131, 99, 6] in BGR -> RGB is [6, 99, 131]
    target_color_bgr = [131, 99, 6]
    target_color_rgb = [6, 99, 131]
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    frames = 64
    for i in range(frames):
        frame = img.copy()
        
        # Calculate current cutoff for wipe effect (bottom to top)
        # At i=0, current_y = max_y + 1 (nothing filled)
        # At i=63, current_y = min_y (all filled)
        progress = i / (frames - 1)
        current_y = int(max_y + 1 - progress * (max_y - min_y + 1))
        
        # Apply fill
        for y in range(161):
            if y >= current_y:
                for x in range(161):
                    if mask5_interior[y, x]:
                        frame[432+y, 832+x] = target_color_bgr
                        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    create_video()
