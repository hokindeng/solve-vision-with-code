import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    # Standard smoothstep
    return t * t * (3 - 2 * t)

def main():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return
        
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image")
        return
        
    bg_color = np.array([240, 245, 245])
    
    # 1. Bounding box of the orange brick in the callout box
    # From previous analysis: x in [31, 289], y in [302, 510]
    # We will use slightly larger just in case, but mask will restrict it
    y_min, y_max = 295, 515
    x_min, x_max = 25, 295
    
    sub_img = img[y_min:y_max, x_min:x_max]
    
    # Non-background pixels
    diff = np.abs(sub_img.astype(np.int32) - bg_color.astype(np.int32))
    non_bg = np.sum(diff, axis=2) > 0
    
    # Pure red mask (the instruction arrow)
    mask_red = cv2.inRange(sub_img, np.array([0, 0, 250]), np.array([10, 10, 255]))
    
    # Sprite is non-background and not pure red
    sprite_mask = non_bg & (mask_red == 0)
    
    # Create the sprite (RGBA)
    sprite = np.zeros((y_max - y_min, x_max - x_min, 4), dtype=np.uint8)
    sprite[:, :, :3] = sub_img
    sprite[:, :, 3] = sprite_mask.astype(np.uint8) * 255
    
    # Create the clean background by erasing the sprite
    clean_bg = img.copy()
    clean_bg[y_min:y_max, x_min:x_max][sprite_mask] = bg_color
    
    # Target translation (found via template matching over the dashed outline)
    dx_total = 514
    dy_total = 482
    
    num_frames = 46
    fps = 16
    frames = []
    
    for i in range(num_frames):
        if i == 0:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            continue
            
        t = i / (num_frames - 1)
        eased_t = ease_in_out(t)
        
        cur_dx = int(round(dx_total * eased_t))
        cur_dy = int(round(dy_total * eased_t))
        
        frame = clean_bg.copy()
        
        # Paste sprite onto frame
        cur_x = x_min + cur_dx
        cur_y = y_min + cur_dy
        
        y1, y2 = cur_y, cur_y + sprite.shape[0]
        x1, x2 = cur_x, cur_x + sprite.shape[1]
        
        # Clip if out of bounds (should not happen with our coordinates, but safe)
        if y1 < 0:
            y1_sprite = -y1
            y1 = 0
        else:
            y1_sprite = 0
            
        if y2 > frame.shape[0]:
            y2_sprite = sprite.shape[0] - (y2 - frame.shape[0])
            y2 = frame.shape[0]
        else:
            y2_sprite = sprite.shape[0]
            
        if x1 < 0:
            x1_sprite = -x1
            x1 = 0
        else:
            x1_sprite = 0
            
        if x2 > frame.shape[1]:
            x2_sprite = sprite.shape[1] - (x2 - frame.shape[1])
            x2 = frame.shape[1]
        else:
            x2_sprite = sprite.shape[1]
            
        alpha_sprite = sprite[y1_sprite:y2_sprite, x1_sprite:x2_sprite, 3] / 255.0
        alpha_bg = 1.0 - alpha_sprite
        
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha_sprite * sprite[y1_sprite:y2_sprite, x1_sprite:x2_sprite, c] +
                                      alpha_bg * frame[y1:y2, x1:x2, c])
                                      
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p', macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()
    
    print(f"Video saved to {out_path}")

if __name__ == '__main__':
    main()
