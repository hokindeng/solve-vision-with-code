import cv2
import numpy as np
import os
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bg_color = np.array([220, 220, 220], dtype=np.uint8)
    
    # Identify active pixels
    mask = np.any(img_rgb != bg_color, axis=-1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    
    # The two solid objects have large areas, while the dashed outlines consist of small segments.
    obj_labels = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            obj_labels.append(i)
            
    assert len(obj_labels) == 2, "Should find exactly two solid objects"
    
    # Identify objects by color
    c1 = np.median(img_rgb[labels == obj_labels[0]], axis=0)
    if c1[0] > 200: # Magenta object has high red channel
        lbl_mag = obj_labels[0]
        lbl_grn = obj_labels[1]
    else:
        lbl_mag = obj_labels[1]
        lbl_grn = obj_labels[0]
        
    # Translations derived by matching the exact center of the object to the center of the outline
    dx_mag, dy_mag = 129.5, -141.5
    dx_grn, dy_grn = 309.0, -215.0
    
    # Background image contains everything EXCEPT the two solid objects (i.e. background color + dashed outlines)
    bg_img = img_rgb.copy()
    bg_img[labels == lbl_mag] = bg_color
    bg_img[labels == lbl_grn] = bg_color
    
    # Create isolated images for each object with the same solid background color
    obj_mag_img = np.full_like(img_rgb, bg_color)
    obj_mag_img[labels == lbl_mag] = img_rgb[labels == lbl_mag]
    
    obj_grn_img = np.full_like(img_rgb, bg_color)
    obj_grn_img[labels == lbl_grn] = img_rgb[labels == lbl_grn]
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', macro_block_size=None, quality=10, pixelformat='yuv420p')
    
    num_frames = 35
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        shift_dx_mag = dx_mag * t
        shift_dy_mag = dy_mag * t
        
        shift_dx_grn = dx_grn * t
        shift_dy_grn = dy_grn * t
        
        # Shift using warpAffine. INTER_LINEAR properly anti-aliases subpixel shifts against the solid background.
        M_mag = np.float32([[1, 0, shift_dx_mag], [0, 1, shift_dy_mag]])
        shifted_mag = cv2.warpAffine(obj_mag_img, M_mag, (img_rgb.shape[1], img_rgb.shape[0]), 
                                     borderValue=(220, 220, 220), flags=cv2.INTER_LINEAR)
                                     
        M_grn = np.float32([[1, 0, shift_dx_grn], [0, 1, shift_dy_grn]])
        shifted_grn = cv2.warpAffine(obj_grn_img, M_grn, (img_rgb.shape[1], img_rgb.shape[0]), 
                                     borderValue=(220, 220, 220), flags=cv2.INTER_LINEAR)
                                     
        # Composite the frame by overlaying the shifted objects onto the background image
        frame = bg_img.copy()
        
        # Only overwrite pixels where the shifted image differs from the background color
        mag_mask = np.any(shifted_mag != bg_color, axis=-1)
        frame[mag_mask] = shifted_mag[mag_mask]
        
        grn_mask = np.any(shifted_grn != bg_color, axis=-1)
        frame[grn_mask] = shifted_grn[grn_mask]
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
