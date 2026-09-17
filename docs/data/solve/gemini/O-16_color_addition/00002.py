import numpy as np
from PIL import Image
import imageio
import os

def main():
    img = Image.open('/app/first_frame.png')
    img_np = np.array(img)
    
    # Extract original ball patches
    y1_start, y1_end = 283, 524
    x1_start, x1_end = 238, 479
    patch1 = img_np[y1_start:y1_end, x1_start:x1_end].copy()
    
    y2_start, y2_end = 602, 843
    x2_start, x2_end = 491, 732
    patch2 = img_np[y2_start:y2_end, x2_start:x2_end].copy()
    
    # Create masks for the balls
    mask1 = np.any(patch1 != [255, 255, 255], axis=-1)
    mask2 = np.any(patch2 != [255, 255, 255], axis=-1)
    
    # Clear background pixels in the patches
    patch1[~mask1] = [0, 0, 0]
    patch2[~mask2] = [0, 0, 0]
    
    # Create the clean background
    bg = img_np.copy()
    bg[y1_start:y1_end, x1_start:x1_end][mask1] = [255, 255, 255]
    bg[y2_start:y2_end, x2_start:x2_end][mask2] = [255, 255, 255]
    
    c1_start_x = 358
    c1_start_y = 403
    
    c2_start_x = 611
    c2_start_y = 722
    
    c_mid_x = 484.5
    c_mid_y = 562.5
    
    num_frames = 80
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        cx1 = c1_start_x + t * (c_mid_x - c1_start_x)
        cy1 = c1_start_y + t * (c_mid_y - c1_start_y)
        
        cx2 = c2_start_x + t * (c_mid_x - c2_start_x)
        cy2 = c2_start_y + t * (c_mid_y - c2_start_y)
        
        tl_x1 = int(round(cx1) - 120)
        tl_y1 = int(round(cy1) - 120)
        
        tl_x2 = int(round(cx2) - 120)
        tl_y2 = int(round(cy2) - 120)
        
        Canvas1 = np.zeros((1024, 1024, 3), dtype=np.uint16)
        Canvas2 = np.zeros((1024, 1024, 3), dtype=np.uint16)
        Mask1 = np.zeros((1024, 1024), dtype=bool)
        Mask2 = np.zeros((1024, 1024), dtype=bool)
        
        Canvas1[tl_y1:tl_y1+241, tl_x1:tl_x1+241][mask1] = patch1[mask1]
        Mask1[tl_y1:tl_y1+241, tl_x1:tl_x1+241] = mask1
        
        Canvas2[tl_y2:tl_y2+241, tl_x2:tl_x2+241][mask2] = patch2[mask2]
        Mask2[tl_y2:tl_y2+241, tl_x2:tl_x2+241] = mask2
        
        combined_balls = np.minimum(Canvas1 + Canvas2, 255).astype(np.uint8)
        combined_mask = Mask1 | Mask2
        
        frame = bg.copy()
        frame[combined_mask] = combined_balls[combined_mask]
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
