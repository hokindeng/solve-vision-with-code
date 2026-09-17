import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    h_img, w_img, _ = img.shape
    
    # Identify non-white pixels as foreground
    mask = np.any(img != 255, axis=-1).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    circles = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        comp_mask = (labels == i)
        
        circles.append({
            'id': i,
            'bbox': (x, y, w, h),
            'area': area,
            'mask': comp_mask,
            'w': w,
            'h': h
        })

    # Sort circles from largest to smallest by width
    circles.sort(key=lambda c: c['w'], reverse=True)

    gap = 30
    total_width = sum(c['w'] for c in circles) + gap * (len(circles) - 1)
    start_x = (w_img - total_width) / 2.0
    
    current_x = start_x
    for c in circles:
        c['target_cx'] = current_x + c['w'] / 2.0
        c['target_cy'] = h_img / 2.0
        c['start_cx'] = float(c['bbox'][0] + c['w'] / 2.0)
        c['start_cy'] = float(c['bbox'][1] + c['h'] / 2.0)
        current_x += c['w'] + gap

    bg = img.copy()
    for c in circles:
        bg[c['mask']] = [255, 255, 255]

    for c in circles:
        x, y, w, h = c['bbox']
        c_img = img[y:y+h, x:x+w].copy()
        c_alpha = c['mask'][y:y+h, x:x+w] # boolean mask
        c['img'] = c_img
        c['alpha'] = c_alpha

    num_frames = 80
    fps = 16
    out_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        e = ease_in_out(t)
        
        frame = bg.copy()
        
        for c in circles:
            cx = c['start_cx'] + (c['target_cx'] - c['start_cx']) * e
            cy = c['start_cy'] + (c['target_cy'] - c['start_cy']) * e
            
            x = int(round(cx - c['w'] / 2.0))
            y = int(round(cy - c['h'] / 2.0))
            
            h, w = c['h'], c['w']
            
            y1, y2 = max(0, y), min(h_img, y + h)
            x1, x2 = max(0, x), min(w_img, x + w)
            
            if y1 < y2 and x1 < x2:
                cy1, cy2 = y1 - y, h - (y + h - y2)
                cx1, cx2 = x1 - x, w - (x + w - x2)
                
                fg = c['img'][cy1:cy2, cx1:cx2]
                alpha_mask = c['alpha'][cy1:cy2, cx1:cx2]
                
                # Apply boolean mask
                frame_roi = frame[y1:y2, x1:x2]
                frame_roi[alpha_mask] = fg[alpha_mask]
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == "__main__":
    main()
