import cv2
import numpy as np
import os
import subprocess

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    h, w, _ = img.shape
    bg_color = img[0, 0]
    
    # Binary mask just to find the bounding boxes
    mask = cv2.inRange(img, bg_color - 5, bg_color + 5)
    mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    
    # We will build a pure background by filling the contours with bg_color
    base_bg = img.copy()
    
    bg_f = bg_color.astype(np.float32)
    
    for i, c in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + bw // 2, y + bh // 2
            
        color = tuple(img[cy, cx].tolist())
        
        # Erase from base background
        shape_mask = np.zeros((h, w), dtype=np.uint8)
        # draw a slightly thicker contour to erase anti-aliasing
        cv2.drawContours(shape_mask, [c], -1, 255, -1)
        kernel = np.ones((5,5), np.uint8)
        shape_mask = cv2.dilate(shape_mask, kernel, iterations=1)
        base_bg[shape_mask > 0] = bg_color
        
        # Extract with alpha
        patch = img[y:y+bh, x:x+bw].astype(np.float32)
        shape_color_f = np.array(color, dtype=np.float32)
        
        dist_patch = np.linalg.norm(patch - bg_f, axis=-1)
        dist_shape = np.linalg.norm(shape_color_f - bg_f)
        
        alpha = dist_patch / (dist_shape + 1e-6)
        alpha = np.clip(alpha, 0, 1)
        
        # Clean up alpha just in case of noise (though there shouldn't be any)
        # Any alpha < 0.01 can be 0
        alpha[alpha < 0.01] = 0
        
        shapes.append({
            'id': i,
            'x': x, 'y': y, 'bw': bw, 'bh': bh,
            'cx': cx, 'cy': cy,
            'area': area,
            'color': color,
            'alpha': alpha,
            'img_f': patch,
            'cx_offset': cx - x,
            'cy_offset': cy - y
        })
        
    # Group by color
    groups = {}
    for s in shapes:
        c = s['color']
        if c not in groups:
            groups[c] = []
        groups[c].append(s)
        
    # Sort groups by color to have a consistent order
    sorted_groups = sorted(groups.keys())
    
    # Sort within groups by size
    ordered_shapes = []
    for c in sorted_groups:
        grp = sorted(groups[c], key=lambda s: s['area'])
        ordered_shapes.extend(grp)
        
    # Calculate target positions
    gap = 40
    total_width = sum(s['bw'] for s in ordered_shapes) + gap * (len(ordered_shapes) - 1)
    
    start_x = (w - total_width) // 2
    current_x = start_x
    
    for s in ordered_shapes:
        s['target_x'] = current_x
        s['target_y'] = (h - s['bh']) // 2
        
        s['target_cx'] = s['target_x'] + s['cx_offset']
        s['target_cy'] = s['target_y'] + s['cy_offset']
        
        current_x += s['bw'] + gap
        
    # Ensure background is perfectly solid in case dilation missed anything
    # Since we know the true bg color, and the original image has no noise,
    # and all shapes were isolated, base_bg should be pure bg_color.
    # We will just use a solid color background for perfection, as the prompt
    # says "keep the background unchanged". If original was solid, we keep it solid.
    base_bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    base_bg_f = base_bg.astype(np.float32)
    
    # Generate frames
    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 96
    
    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        progress = ease_in_out(t)
        
        frame_f = base_bg_f.copy()
        
        # Sort shapes by area descending so smaller shapes draw on top during overlap
        draw_shapes = sorted(shapes, key=lambda s: s['area'], reverse=True)
        
        for s in draw_shapes:
            curr_cx = s['cx'] + (s['target_cx'] - s['cx']) * progress
            curr_cy = s['cy'] + (s['target_cy'] - s['cy']) * progress
            
            curr_x = int(curr_cx - s['cx_offset'])
            curr_y = int(curr_cy - s['cy_offset'])
            
            y1, y2 = curr_y, curr_y + s['bh']
            x1, x2 = curr_x, curr_x + s['bw']
            
            # Clipping
            img_y1 = max(0, y1)
            img_y2 = min(h, y2)
            img_x1 = max(0, x1)
            img_x2 = min(w, x2)
            
            if img_y1 >= img_y2 or img_x1 >= img_x2:
                continue
                
            patch_y1 = img_y1 - y1
            patch_y2 = s['bh'] - (y2 - img_y2)
            patch_x1 = img_x1 - x1
            patch_x2 = s['bw'] - (x2 - img_x2)
            
            patch_alpha = s['alpha'][patch_y1:patch_y2, patch_x1:patch_x2, None]
            fg = s['img_f'][patch_y1:patch_y2, patch_x1:patch_x2]
            bg = frame_f[img_y1:img_y2, img_x1:img_x2]
            
            frame_f[img_y1:img_y2, img_x1:img_x2] = fg * patch_alpha + bg * (1 - patch_alpha)
            
        frame = frame_f.astype(np.uint8)
        cv2.imwrite(f'{frames_dir}/frame_{frame_idx:04d}.png', frame)
        
    # Make video
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)
    
if __name__ == '__main__':
    main()
