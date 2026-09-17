import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    bg_color = img[0, 0]
    
    # Calculate difference from background to find shape pixels
    diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16))
    mask_full = np.any(diff > 0, axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        ratio = area / (w * h)
        # Ratio ~ 1.0 for squares, ~ 0.5 for diamonds
        shape_type = 'square' if ratio > 0.8 else 'diamond'
        
        patch = img[y:y+h, x:x+w].copy()
        patch_mask = mask_full[y:y+h, x:x+w] > 0
        
        shapes.append({
            'id': i,
            'rect': (x, y, w, h),
            'area': area,
            'type': shape_type,
            'start_x': x + w / 2,
            'start_y': y + h / 2,
            'w': w,
            'h': h,
            'patch': patch,
            'patch_mask': patch_mask
        })
    
    squares = [s for s in shapes if s['type'] == 'square']
    diamonds = [s for s in shapes if s['type'] == 'diamond']
    
    # Sort within groups by size (width)
    squares.sort(key=lambda s: s['w'])
    diamonds.sort(key=lambda s: s['w'])
    
    # Put diamonds first, then squares
    final_order = diamonds + squares
    
    # Calculate final positions
    total_width = sum(s['w'] for s in final_order)
    remaining_space = 1024 - total_width
    spacing = remaining_space / (len(final_order) + 1)
    
    current_x = spacing
    for s in final_order:
        s['end_x'] = current_x + s['w'] / 2
        s['end_y'] = 512
        current_x += s['w'] + spacing
    
    # Draw larger shapes first so smaller shapes are drawn on top if they overlap
    shapes.sort(key=lambda s: s['area'], reverse=True)
    
    os.makedirs('/app/output', exist_ok=True)
    # H.264, yuv420p, 1024x1024, 16 fps
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None
    )
    
    clean_bg = np.full_like(img, bg_color)
    
    frames = 96
    for f in range(frames):
        t = f / (frames - 1)
        # ease in out smoothstep
        t_ease = t * t * (3 - 2 * t)
        
        frame = clean_bg.copy()
        
        for s in shapes:
            cx = (1 - t_ease) * s['start_x'] + t_ease * s['end_x']
            cy = (1 - t_ease) * s['start_y'] + t_ease * s['end_y']
            
            left = int(round(cx - s['w'] / 2))
            top = int(round(cy - s['h'] / 2))
            
            patch = s['patch']
            patch_mask = s['patch_mask']
            
            roi = frame[top:top+s['h'], left:left+s['w']]
            frame[top:top+s['h'], left:left+s['w']] = np.where(patch_mask[:, :, None], patch, roi)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
