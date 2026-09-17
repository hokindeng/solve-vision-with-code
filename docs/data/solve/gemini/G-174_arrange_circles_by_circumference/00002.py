import cv2
import numpy as np
import imageio
import math
import os

def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    h, w_img, _ = img.shape
    
    # Background color is white
    non_white = np.any(img != 255, axis=-1).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_white, connectivity=4)
    
    circles = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h_bb = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Use bounding box center to ensure frame 0 matches perfectly
        cx = x + w / 2.0
        cy = y + h_bb / 2.0
        
        # Extract the patch and mask
        patch = img[y:y+h_bb, x:x+w].copy()
        mask = (labels[y:y+h_bb, x:x+w] == i)
        
        circles.append({
            'label': i,
            'w': w,
            'h': h_bb,
            'start_cx': cx,
            'start_cy': cy,
            'patch': patch,
            'mask': mask
        })
        
    # Sort by width (diameter), largest to smallest
    circles.sort(key=lambda c: c['w'], reverse=True)
    
    gap = 20
    total_width = sum(c['w'] for c in circles) + gap * (len(circles) - 1)
    
    current_x = (w_img - total_width) / 2.0
    target_cy = h / 2.0
    
    for c in circles:
        c['target_cx'] = current_x + c['w'] / 2.0
        c['target_cy'] = target_cy
        current_x += c['w'] + gap

    num_frames = 80
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, macro_block_size=1)
    
    for frame_idx in range(num_frames):
        # Progress 0 to 1
        t = frame_idx / max(1, (num_frames - 1))
        eased_t = ease_in_out(t)
        
        # Create a blank white image
        frame = np.full((h, w_img, 3), 255, dtype=np.uint8)
        
        for c in circles:
            cx = c['start_cx'] + (c['target_cx'] - c['start_cx']) * eased_t
            cy = c['start_cy'] + (c['target_cy'] - c['start_cy']) * eased_t
            
            # Top-left corner
            tl_x = int(round(cx - c['w'] / 2.0))
            tl_y = int(round(cy - c['h'] / 2.0))
            
            # Use numpy masking
            patch = c['patch']
            mask = c['mask']
            
            # Calculate bounds
            y1 = max(0, tl_y)
            y2 = min(h, tl_y + c['h'])
            x1 = max(0, tl_x)
            x2 = min(w_img, tl_x + c['w'])
            
            if y1 < y2 and x1 < x2:
                # Corresponding bounds in the patch
                py1 = y1 - tl_y
                py2 = py1 + (y2 - y1)
                px1 = x1 - tl_x
                px2 = px1 + (x2 - x1)
                
                patch_view = patch[py1:py2, px1:px2]
                mask_view = mask[py1:py2, px1:px2]
                frame_view = frame[y1:y2, x1:x2]
                
                # Apply mask to the frame
                # Since mask_view is a boolean array, we use it to index the views
                frame_view[mask_view] = patch_view[mask_view]
                            
        # RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
