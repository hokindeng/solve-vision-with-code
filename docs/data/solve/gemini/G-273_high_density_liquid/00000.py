import cv2
import numpy as np
import imageio
from PIL import Image

def main():
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    
    # Object properties
    y_start = 102
    obj_h = 49
    obj_w = 77
    x_coords = [125, 369, 613, 857]
    
    # End positions for each object
    # Cup 1-3 sink: bottom at y=855 -> y_end = 855 - 49 + 1 = 807
    # Cup 4 floats: surface at y=611 -> y_end = 611 - 49//2 = 587
    y_ends = [807, 807, 807, 587]
    
    # Extract object patch and mask from the first object
    patch = img[y_start:y_start+obj_h, x_coords[0]:x_coords[0]+obj_w].copy()
    mask = (patch != [255, 255, 255]).any(axis=-1)
    
    # Create a background image where the initial objects are erased (filled with white)
    bg = img.copy()
    for x in x_coords:
        bg[y_start:y_start+obj_h, x:x+obj_w][mask] = [255, 255, 255]
        
    num_frames = 80
    fps = 16
    start_fall_frame = 10
    end_fall_frame = 70
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, macro_block_size=1)
    
    for frame in range(num_frames):
        frame_img = bg.copy()
        
        # Calculate t in [0, 1]
        t = 0.0
        if frame > start_fall_frame:
            t = (frame - start_fall_frame) / (end_fall_frame - start_fall_frame)
            t = min(1.0, t)
        
        # Ease in (accelerate) then linear, or just ease-in-out
        # Let's use a simple smoothstep for better animation
        # smoothstep: t * t * (3 - 2 * t)
        ease_t = t * t * (3 - 2 * t)
        
        for i, x in enumerate(x_coords):
            y_curr = int(y_start + (y_ends[i] - y_start) * ease_t)
            
            # Place the object onto frame_img using the mask
            roi = frame_img[y_curr:y_curr+obj_h, x:x+obj_w]
            # Replace pixels where mask is True
            np.copyto(roi, patch, where=mask[:, :, None])
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
