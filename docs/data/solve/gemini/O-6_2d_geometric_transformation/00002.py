import cv2
import numpy as np
import imageio
import os

def main():
    # Load original frame
    img = cv2.imread('/app/first_frame.png')
    
    # Polygon masks
    mask_poly = np.all(img == [119, 139, 60], axis=-1) | np.all(img == [50, 50, 50], axis=-1)
    
    # Image containing only the polygon (rest is 0)
    poly_img = np.zeros_like(img)
    poly_img[mask_poly] = img[mask_poly]
    
    # Background (remove polygon, replace with background color)
    bg = img.copy()
    bg[mask_poly] = [240, 240, 240]
    
    # Center marker
    mask_center = np.all(img == [0, 0, 0], axis=-1) | np.all(img == [255, 255, 255], axis=-1)
    center_img = np.zeros_like(img)
    center_img[mask_center] = img[mask_center]
    
    cx, cy = 793, 214
    max_angle = 75.82
    num_frames = 70
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for i in range(num_frames):
        # Progress from 0 to 1
        t = i / (num_frames - 1)
        angle = t * max_angle
        
        # Warp polygon mask and image using nearest neighbor
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_mask = cv2.warpAffine(mask_poly.astype(np.uint8), M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        rotated_poly = cv2.warpAffine(poly_img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        
        # Composite
        comp = bg.copy()
        comp[rotated_mask > 0] = rotated_poly[rotated_mask > 0]
        
        # Keep center marker on top
        comp[mask_center] = center_img[mask_center]
        
        # Convert BGR to RGB for imageio
        comp_rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        
        writer.append_data(comp_rgb)
    
    writer.close()
    print("Video generated.")

if __name__ == '__main__':
    main()
