import cv2
import numpy as np
import os
import subprocess

def ease(t):
    return t * t * (3 - 2 * t)

def composite_alpha(bg, fg, x, y, alpha=1.0):
    if alpha <= 0:
        return bg
    
    fh, fw = fg.shape[:2]
    fg_rgb = fg[:, :, :3]
    fg_a = (fg[:, :, 3] / 255.0) * alpha
    bg_patch = bg[y:y+fh, x:x+fw]
    fg_a_3 = np.expand_dims(fg_a, axis=2)
    blended = fg_rgb * fg_a_3 + bg_patch * (1 - fg_a_3)
    bg[y:y+fh, x:x+fw] = blended
    return bg

def main():
    img = cv2.imread('/app/first_frame.png')
    
    box_xs = [44, 149, 254, 359, 464, 569, 674, 779, 884]
    y_top = 464
    y_bot = 561
    
    # Extract items
    items = []
    rel_x = []
    rel_y = []
    
    for i in range(4):
        x = box_xs[i]
        patch = img[y_top+2:y_bot-2, x+2:x+95].copy()
        mask = np.any(patch != 255, axis=-1).astype(np.uint8) * 255
        x_min, y_min, w, h = cv2.boundingRect(mask)
        
        item = patch[y_min:y_min+h, x_min:x_min+w]
        item_mask = mask[y_min:y_min+h, x_min:x_min+w]
        
        rgba = cv2.cvtColor(item, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = item_mask
        
        items.append(rgba)
        rel_x.append(2 + x_min)
        rel_y.append(2 + y_min)
        
    item0_rgba = items[0]
    
    # Get clean background by erasing boxes 0, 1, 2, 3 using box 4
    clean_bg = img.copy()
    box4 = img[464:561, 464:561]
    
    for i in range(4):
        x = box_xs[i]
        clean_bg[464:561, x:x+97] = box4
        
    out_dir = '/app/frames'
    os.makedirs(out_dir, exist_ok=True)
    
    for f in range(76):
        frame = clean_bg.copy()
        
        # M0 (Item 0)
        if f < 2:
            x = box_xs[0]
        elif f <= 10:
            t = (f - 2) / 8.0
            x = box_xs[0] + (box_xs[1] - box_xs[0]) * ease(t)
        else:
            x = box_xs[1]
        composite_alpha(frame, items[0], int(x) + rel_x[0], 464 + rel_y[0])
        
        # M1 (Item 1)
        if f < 2:
            x = box_xs[1]
        elif f <= 10:
            t = (f - 2) / 8.0
            x = box_xs[1] + (box_xs[2] - box_xs[1]) * ease(t)
        else:
            x = box_xs[2]
        composite_alpha(frame, items[1], int(x) + rel_x[1], 464 + rel_y[1])
        
        # M2 (Item 2)
        if f < 2:
            x = box_xs[2]
        elif f <= 10:
            t = (f - 2) / 8.0
            x = box_xs[2] + (box_xs[3] - box_xs[2]) * ease(t)
        else:
            x = box_xs[3]
        composite_alpha(frame, items[2], int(x) + rel_x[2], 464 + rel_y[2])
        
        # M3 (Item 3)
        if f < 2:
            x = box_xs[3]
        elif f <= 10:
            t = (f - 2) / 8.0
            x = box_xs[3] + (box_xs[4] - box_xs[3]) * ease(t)
        elif f < 27:
            x = box_xs[4]
        elif f <= 35:
            t = (f - 27) / 8.0
            x = box_xs[4] + (box_xs[5] - box_xs[4]) * ease(t)
        elif f < 52:
            x = box_xs[5]
        elif f <= 60:
            t = (f - 52) / 8.0
            x = box_xs[5] + (box_xs[6] - box_xs[5]) * ease(t)
        else:
            x = box_xs[6]
        composite_alpha(frame, items[3], int(x) + rel_x[3], 464 + rel_y[3])
        
        # F1 (Item 0)
        if f >= 12:
            alpha = min(1.0, (f - 12) / 8.0) if f <= 20 else 1.0
            composite_alpha(frame, item0_rgba, box_xs[0] + rel_x[0], 464 + rel_y[0], alpha)
            
        # F2 (Item 0)
        if f >= 37:
            alpha = min(1.0, (f - 37) / 8.0) if f <= 45 else 1.0
            composite_alpha(frame, item0_rgba, box_xs[4] + rel_x[0], 464 + rel_y[0], alpha)
            
        # F3 (Item 0)
        if f >= 62:
            alpha = min(1.0, (f - 62) / 8.0) if f <= 70 else 1.0
            composite_alpha(frame, item0_rgba, box_xs[5] + rel_x[0], 464 + rel_y[0], alpha)
            
        cv2.imwrite(f"{out_dir}/frame_{f:04d}.png", frame)
        
    os.makedirs('/app/output', exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{out_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)
    
if __name__ == '__main__':
    main()
