import cv2
import numpy as np
import imageio

def ease_in_out_quad(t):
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2)**2 / 2

def main():
    img_orig = cv2.imread('/app/first_frame.png')
    
    # Dynamically find objects and baseline
    mask = np.any(img_orig < 255, axis=2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    baseline = None
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10:
            obj_img = img_orig[y:y+h, x:x+w].copy()
            
            # create a solid mask from the contour
            solid_full = np.zeros(img_orig.shape[:2], dtype=bool)
            cv2.drawContours(np.uint8(solid_full)*255, [c], -1, 255, -1)
            # Wait, cv2.drawContours needs a uint8 image
            solid_uint8 = np.zeros(img_orig.shape[:2], dtype=np.uint8)
            cv2.drawContours(solid_uint8, [c], -1, 255, -1)
            solid_mask_crop = (solid_uint8[y:y+h, x:x+w] == 255)
            
            strict_mask = np.any(obj_img < 255, axis=2)
            
            # combine masks
            obj_mask = solid_mask_crop | strict_mask
            
            objects.append({
                'img': obj_img,
                'mask': obj_mask,
                'w': w, 'h': h,
                'start_x': x, 'start_y': y
            })
        else:
            baseline = (x, y, w, h)
            
    # Create base image (background + baseline)
    base_img = img_orig.copy()
    for obj in objects:
        x, y, w, h = obj['start_x'], obj['start_y'], obj['w'], obj['h']
        base_img[y:y+h, x:x+w] = 255
        
    # 2. Determine final positions
    objects.sort(key=lambda o: o['w'] * o['h'], reverse=True)
    
    # Baseline specs
    baseline_x, baseline_y, baseline_w, baseline_h = baseline
    
    total_obj_w = sum(o['w'] for o in objects)
    remaining_space = baseline_w - total_obj_w
    space_between = remaining_space // (len(objects) + 1)
    
    current_x = baseline_x + space_between
    for obj in objects:
        obj['end_x'] = current_x
        # Align bottom of object to top of baseline
        obj['end_y'] = baseline_y - obj['h']
        current_x += obj['w'] + space_between
        
    # 3. Generate frames
    num_frames = 40
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    for i in range(num_frames):
        t = i / max(1, (num_frames - 1))
        progress = ease_in_out_quad(t)
        
        frame = base_img.copy()
        
        for obj in objects:
            curr_x = int(round(obj['start_x'] + (obj['end_x'] - obj['start_x']) * progress))
            curr_y = int(round(obj['start_y'] + (obj['end_y'] - obj['start_y']) * progress))
            
            h, w = obj['h'], obj['w']
            
            mask_obj = obj['mask']
            frame[curr_y:curr_y+h, curr_x:curr_x+w][mask_obj] = obj['img'][mask_obj]
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
