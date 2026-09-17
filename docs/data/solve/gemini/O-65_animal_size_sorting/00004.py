import cv2
import numpy as np
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 255
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    baseline = None
    animals = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 900:  # Baseline
            baseline = (x, y, w, h)
        else:
            # animal dict: orig_bbox, img_crop, mask_crop
            anim_img = img[y:y+h, x:x+w].copy()
            solid_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(solid_mask, [c], -1, 255, -1)
            anim_mask = (solid_mask[y:y+h, x:x+w] > 0)
            
            animals.append({
                'bbox': (x, y, w, h),
                'img': anim_img,
                'mask': anim_mask,
                'area': w * h
            })
            
    # Sort animals by area descending (largest to smallest)
    animals.sort(key=lambda a: a['area'], reverse=True)
    
    # Calculate target positions
    total_w = sum(a['bbox'][2] for a in animals)
    base_x, base_y, base_w, base_h = baseline
    
    gap = (base_w - total_w) / (len(animals) - 1) if len(animals) > 1 else 0
    
    curr_x = base_x
    for a in animals:
        w, h = a['bbox'][2], a['bbox'][3]
        target_x = int(round(curr_x))
        # The animal will sit perfectly on top of the baseline
        target_y = base_y - h
        a['target_bbox'] = (target_x, target_y, w, h)
        curr_x += w + gap

    frames = []
    num_frames = 40
    
    # Pre-calculate pure background (white + baseline)
    bg = np.full_like(img, 255)
    bg[base_y:base_y+base_h, base_x:base_x+base_w] = img[base_y:base_y+base_h, base_x:base_x+base_w]
    
    for i in range(num_frames):
        frame = bg.copy()
        
        # interpolate positions (0.0 to 1.0)
        t = i / (num_frames - 1) if num_frames > 1 else 1.0
            
        for a in animals:
            sx, sy, w, h = a['bbox']
            tx, ty, _, _ = a['target_bbox']
            
            cur_x = int(round(sx + (tx - sx) * t))
            cur_y = int(round(sy + (ty - sy) * t))
            
            # Place animal on frame
            anim_mask = a['mask']
            frame_roi = frame[cur_y:cur_y+h, cur_x:cur_x+w]
            np.copyto(frame_roi, a['img'], where=anim_mask[:, :, None])
            
        frames.append(frame)
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        # Convert BGR to RGB for imageio
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()

if __name__ == '__main__':
    solve()
