import cv2
import numpy as np
import imageio
import os

def solve():
    # 1. Read first frame
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # work in RGB for imageio
    
    bg_color = img[0, 0].copy()
    
    # 2. Extract shapes
    # Create mask of non-background pixels
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    shapes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Get shape crop
        shape_crop = img[y:y+h, x:x+w].copy()
        shape_mask = (labels[y:y+h, x:x+w] == i)
        
        # Determine dominant color (ignoring background within the bounding box)
        pixels = shape_crop[shape_mask]
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        dom_color = unique_colors[np.argmax(counts)]
        
        shapes.append({
            'id': i,
            'x': float(x),
            'y': float(y),
            'w': w,
            'h': h,
            'crop': shape_crop,
            'mask': shape_mask,
            'color': tuple(dom_color.tolist())
        })
        
    # 3. Group and sort
    groups = {}
    for s in shapes:
        c = s['color']
        if c not in groups:
            groups[c] = []
        groups[c].append(s)
        
    group_list = list(groups.values())
    group_list.sort(key=lambda g: sum(s['x'] for s in g) / len(g))
    
    for g in group_list:
        g.sort(key=lambda s: s['w'])
        
    final_shapes = []
    for g in group_list:
        final_shapes.extend(g)
        
    # 4. Calculate target positions
    gap = 30
    total_width = sum(s['w'] for s in final_shapes) + gap * (len(final_shapes) - 1)
    
    current_x = (1024 - total_width) // 2
    
    for s in final_shapes:
        s['target_x'] = float(current_x)
        s['target_y'] = float(512 - s['h'] // 2)
        current_x += s['w'] + gap
        
    # 5. Generate video
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    frames = 96
    
    def smoothstep(t):
        if t <= 0: return 0.0
        if t >= 1: return 1.0
        return t * t * (3.0 - 2.0 * t)
        
    for f in range(frames):
        t = f / (frames - 1)
        ease = smoothstep(t)
        
        frame = np.empty((1024, 1024, 3), dtype=np.uint8)
        frame[:] = bg_color
        
        for s in final_shapes:
            cur_x = int(round(s['x'] + (s['target_x'] - s['x']) * ease))
            cur_y = int(round(s['y'] + (s['target_y'] - s['y']) * ease))
            
            h, w = s['h'], s['w']
            
            dst_y1, dst_y2 = cur_y, cur_y + h
            dst_x1, dst_x2 = cur_x, cur_x + w
            
            src_y1, src_y2 = 0, h
            src_x1, src_x2 = 0, w
            
            if dst_x1 < 0:
                src_x1 -= dst_x1
                dst_x1 = 0
            if dst_y1 < 0:
                src_y1 -= dst_y1
                dst_y1 = 0
            if dst_x2 > 1024:
                src_x2 -= (dst_x2 - 1024)
                dst_x2 = 1024
            if dst_y2 > 1024:
                src_y2 -= (dst_y2 - 1024)
                dst_y2 = 1024
                
            if src_x1 >= src_x2 or src_y1 >= src_y2:
                continue
                
            crop_slice = s['crop'][src_y1:src_y2, src_x1:src_x2]
            mask_slice = s['mask'][src_y1:src_y2, src_x1:src_x2]
            frame_slice = frame[dst_y1:dst_y2, dst_x1:dst_x2]
            
            frame_slice[mask_slice] = crop_slice[mask_slice]
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
