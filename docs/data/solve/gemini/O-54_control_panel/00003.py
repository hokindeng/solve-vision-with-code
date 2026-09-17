import cv2
import numpy as np
import subprocess
import os

def ease_in_out(t):
    return 0.5 - 0.5 * np.cos(t * np.pi)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Pre-extract lever regions and median background
    clean_slot_0 = img[600:750, 50:350].copy()
    clean_slot_1 = img[600:750, 358:658].copy()
    clean_slot_2 = img[600:750, 666:966].copy()
    
    median_bg = np.median([clean_slot_0, clean_slot_1, clean_slot_2], axis=0).astype(np.uint8)
    
    # Knob from Unit 1
    knob_patch = clean_slot_1[38:38+57, 172:172+57]
    mask1 = (np.sum(cv2.absdiff(clean_slot_1, median_bg), axis=2) > 0).astype(np.uint8)
    knob_mask = mask1[38:38+57, 172:172+57]
    
    # Light patches
    light1 = img[150:350, 358:658].copy()
    light0_orig = img[150:350, 50:350].copy()
    light2_orig = img[150:350, 666:966].copy()
    
    os.makedirs('/app/output/frames', exist_ok=True)
    
    num_frames = 24
    for f in range(num_frames):
        t = f / (num_frames - 1)
        t_ease = ease_in_out(t)
        
        out = img.copy()
        
        # --- LEVERS ---
        # Unit 0
        base0 = clean_slot_0.copy()
        # Erase knob at x=19
        base0[38:38+57, 19:19+57] = median_bg[38:38+57, 19:19+57]
        # Draw knob at new x
        x0 = int(19 + (172 - 19) * t_ease)
        for c in range(3):
            base0[38:38+57, x0:x0+57, c] = np.where(knob_mask, knob_patch[:,:,c], base0[38:38+57, x0:x0+57, c])
        out[600:750, 50:350] = base0
        
        # Unit 2
        base2 = clean_slot_2.copy()
        # Erase knob at x=94
        base2[38:38+57, 94:94+57] = median_bg[38:38+57, 94:94+57]
        # Draw knob at new x
        x2 = int(94 + (172 - 94) * t_ease)
        for c in range(3):
            base2[38:38+57, x2:x2+57, c] = np.where(knob_mask, knob_patch[:,:,c], base2[38:38+57, x2:x2+57, c])
        out[600:750, 666:966] = base2
        
        # --- LIGHTS ---
        # Unit 0
        blend0 = cv2.addWeighted(light0_orig, 1 - t_ease, light1, t_ease, 0)
        out[150:350, 50:350] = blend0
        
        # Unit 2
        blend2 = cv2.addWeighted(light2_orig, 1 - t_ease, light1, t_ease, 0)
        out[150:350, 666:966] = blend2
        
        cv2.imwrite(f'/app/output/frames/frame_{f:04d}.png', out)
        
    # Generate video
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/output/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)
    
if __name__ == '__main__':
    main()
