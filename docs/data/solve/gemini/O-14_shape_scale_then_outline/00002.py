import os
import cv2
import numpy as np
import imageio

def solve():
    img_first = cv2.imread('/app/first_frame.png')
    img_first = cv2.cvtColor(img_first, cv2.COLOR_BGR2RGB)
    
    # 1. Extract the input mask
    # Green is RGB [89, 153, 76]
    mask_in = np.all(img_first == [89, 153, 76], axis=-1).astype(np.uint8)
    roi_mask = mask_in[628:727, 121:226]
    
    # 2. Precompute the scaled mask (Step 1 target)
    w_scaled = int(105 * 0.78) # 81
    h_scaled = int(99 * 0.78)  # 77
    scaled_mask = cv2.resize(roi_mask, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)
    
    # 3. Precompute the filled mask (Step 2 target)
    filled_mask = scaled_mask.copy()
    cv2.floodFill(filled_mask, None, (w_scaled//2, h_scaled//2), 1)
    
    def erase_qmarks(frame):
        roi1 = frame[640:715, 450:520]
        m1 = ~np.all(roi1 == [255, 255, 255], axis=-1) & ~np.all(roi1 == [89, 153, 76], axis=-1) & ~np.all(roi1 == [0,0,0], axis=-1)
        frame[640:715, 450:520][m1] = [255, 255, 255]
        
        roi2 = frame[640:715, 755:825]
        m2 = ~np.all(roi2 == [255, 255, 255], axis=-1) & ~np.all(roi2 == [89, 153, 76], axis=-1) & ~np.all(roi2 == [0,0,0], axis=-1)
        frame[640:715, 755:825][m2] = [255, 255, 255]
        
    def draw_centered(frame, mask, cx, cy, color=[89, 153, 76]):
        h, w = mask.shape
        top = cy - h // 2
        left = cx - w // 2
        frame_roi = frame[top:top+h, left:left+w]
        frame_roi[mask > 0] = color
        
    frames = []
    frames.append(img_first.copy())
    
    for f in range(1, 16):
        frame = img_first.copy()
        erase_qmarks(frame)
        
        if f <= 7:
            # Step 1: Scale change at Middle slot
            progress = (f - 1) / 6.0 # 0.0 to 1.0
            s = 1.0 - 0.22 * progress
            cw = max(1, int(105 * s))
            ch = max(1, int(99 * s))
            curr_mask = cv2.resize(roi_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
            draw_centered(frame, curr_mask, 479, 678)
        else:
            # Step 1 is complete, Middle slot stays scaled
            draw_centered(frame, scaled_mask, 479, 678)
            
            # Step 2: Fill-to-outline (sweeping fill) at Output slot
            progress = (f - 8) / 7.0 # 0.0 to 1.0
            limit_x = int(w_scaled * progress)
            
            out_mask = np.zeros_like(scaled_mask)
            for x in range(w_scaled):
                if x <= limit_x:
                    out_mask[:, x] = filled_mask[:, x]
                else:
                    out_mask[:, x] = scaled_mask[:, x]
                    
            draw_centered(frame, out_mask, 786, 678)
            
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
