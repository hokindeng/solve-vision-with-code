import cv2
import numpy as np
import imageio
import os

def solve():
    image = cv2.imread('/app/first_frame.png')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    objects = []
    markers = []
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # Bounding box center
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        # Get color at centroid
        c_x, c_y = int(centroids[i][0]), int(centroids[i][1])
        color = tuple(image_rgb[c_y, c_x])
        
        comp_mask = (labels == i)
        
        info = {
            'label': i,
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'color': color,
            'mask': comp_mask
        }
        
        if area > 1000:
            objects.append(info)
        else:
            markers.append(info)
            
    # Create background by removing objects
    background = image_rgb.copy()
    for obj in objects:
        background[obj['mask']] = [255, 255, 255]
        
    # Match objects to markers by color
    animations = []
    for obj in objects:
        for marker in markers:
            if obj['color'] == marker['color']:
                start_x, start_y = obj['bbox'][0], obj['bbox'][1]
                
                # Target top-left corner
                target_x = int(marker['center'][0] - obj['bbox'][2] / 2.0)
                target_y = int(marker['center'][1] - obj['bbox'][3] / 2.0)
                
                animations.append({
                    'obj': obj,
                    'start': (start_x, start_y),
                    'end': (target_x, target_y)
                })
                break
                
    # Generate frames
    num_frames = 48
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for i in range(num_frames):
        frame = background.copy()
        
        # Calculate t from 0.0 to 1.0
        t = i / (num_frames - 1) if num_frames > 1 else 1.0
        
        for anim in animations:
            obj = anim['obj']
            mask = obj['mask']
            
            # Extract object pixels
            # Since the object bounding box might be large, we can just use the mask
            # But the mask moves. Better: extract object from original image.
            # We can find the relative offsets within the bounding box
            x, y, w, h = obj['bbox']
            obj_img = image_rgb[y:y+h, x:x+w]
            obj_mask = mask[y:y+h, x:x+w]
            
            current_x = int(round(anim['start'][0] + t * (anim['end'][0] - anim['start'][0])))
            current_y = int(round(anim['start'][1] + t * (anim['end'][1] - anim['start'][1])))
            
            # Paste into frame
            # We need to handle bounds if objects move outside, but here they stay inside
            for dy in range(h):
                for dx in range(w):
                    if obj_mask[dy, dx]:
                        fy = current_y + dy
                        fx = current_x + dx
                        if 0 <= fy < frame.shape[0] and 0 <= fx < frame.shape[1]:
                            frame[fy, fx] = obj_img[dy, dx]
                            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
