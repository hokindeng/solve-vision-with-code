import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hardcoded colors observed in the first frame
    color1 = np.array([108, 73, 107])
    color2 = np.array([54, 112, 92])
    color_black = np.array([0, 0, 0])

    mask_b1 = (img_rgb == color1).all(axis=-1)
    mask_b2 = (img_rgb == color2).all(axis=-1)

    # find outlines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((img_rgb == color_black).all(axis=-1).astype(np.uint8))
    
    out1_mask = np.zeros_like(mask_b1)
    out2_mask = np.zeros_like(mask_b2)
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        if cx < img_rgb.shape[1] / 2: # left side, ball 1
            out1_mask = (labels == i)
        else: # right side, ball 2
            out2_mask = (labels == i)

    D1 = mask_b1 | out1_mask
    D2 = mask_b2 | out2_mask

    y1, x1 = np.where(D1)
    crop1 = img_rgb[y1.min():y1.max()+1, x1.min():x1.max()+1]
    shape_mask1 = D1[y1.min():y1.max()+1, x1.min():x1.max()+1]
    crop1_colors = np.zeros_like(crop1)
    crop1_colors[shape_mask1] = crop1[shape_mask1]

    y2, x2 = np.where(D2)
    crop2 = img_rgb[y2.min():y2.max()+1, x2.min():x2.max()+1]
    shape_mask2 = D2[y2.min():y2.max()+1, x2.min():x2.max()+1]
    crop2_colors = np.zeros_like(crop2)
    crop2_colors[shape_mask2] = crop2[shape_mask2]

    # Background canvas
    bg = img_rgb.copy()
    bg[D1 | D2] = [255, 255, 255]

    frames = []
    
    start_cx1 = (x1.min() + x1.max()) / 2.0
    start_cy1 = (y1.min() + y1.max()) / 2.0
    
    start_cx2 = (x2.min() + x2.max()) / 2.0
    start_cy2 = (y2.min() + y2.max()) / 2.0
    
    end_cx = (start_cx1 + start_cx2) / 2.0
    end_cy = (start_cy1 + start_cy2) / 2.0

    num_frames = 80
    H, W = img_rgb.shape[:2]
    
    h1, w1 = crop1_colors.shape[:2]
    h2, w2 = crop2_colors.shape[:2]

    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        cx1 = start_cx1 + t * (end_cx - start_cx1)
        cy1 = start_cy1 + t * (end_cy - start_cy1)
        
        cx2 = start_cx2 + t * (end_cx - start_cx2)
        cy2 = start_cy2 + t * (end_cy - start_cy2)
        
        tx1 = int(round(cx1 - w1 / 2.0 + 0.5)) if w1 % 2 != 0 else int(round(cx1 - w1 / 2.0))
        ty1 = int(round(cy1 - h1 / 2.0 + 0.5)) if h1 % 2 != 0 else int(round(cy1 - h1 / 2.0))
        
        tx2 = int(round(cx2 - w2 / 2.0 + 0.5)) if w2 % 2 != 0 else int(round(cx2 - w2 / 2.0))
        ty2 = int(round(cy2 - h2 / 2.0 + 0.5)) if h2 % 2 != 0 else int(round(cy2 - h2 / 2.0))
        
        c1 = np.zeros((H, W, 3), dtype=np.uint16)
        a1 = np.zeros((H, W), dtype=bool)
        c1[ty1:ty1+h1, tx1:tx1+w1] = crop1_colors
        a1[ty1:ty1+h1, tx1:tx1+w1] = shape_mask1
        
        c2 = np.zeros((H, W, 3), dtype=np.uint16)
        a2 = np.zeros((H, W), dtype=bool)
        c2[ty2:ty2+h2, tx2:tx2+w2] = crop2_colors
        a2[ty2:ty2+h2, tx2:tx2+w2] = shape_mask2
        
        a_combined = a1 | a2
        # Additive light mixing
        c_combined = np.minimum(255, c1 + c2).astype(np.uint8)
        
        frame = bg.copy()
        frame[a_combined] = c_combined[a_combined]
        
        frames.append(frame)

    imageio.mimwrite(
        '/app/output/video.mp4', 
        frames, 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None,
        quality=10
    )

if __name__ == '__main__':
    main()
