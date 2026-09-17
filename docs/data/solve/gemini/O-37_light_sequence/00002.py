import cv2
import numpy as np
import imageio
import os

def main():
    # Ensure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract patches from first frame
    # Y center is 511, X centers are 170, 307, 443, 580, 716, 853
    # We use a 120x120 patch which fully encloses the light and glow
    patch_size = 60
    y_c = 511
    
    centers = [170, 307, 443, 580, 716, 853]
    
    # Light 0 is OFF, Light 5 is ON
    off_patch = img_rgb[y_c - patch_size : y_c + patch_size, centers[0] - patch_size : centers[0] + patch_size].astype(np.float32)
    on_patch = img_rgb[y_c - patch_size : y_c + patch_size, centers[5] - patch_size : centers[5] + patch_size].astype(np.float32)
    
    # Initial and target states for each light
    # 0 = OFF, 1 = ON
    initial_states = [0, 0, 0, 0, 0, 1]
    target_states  = [0, 1, 1, 1, 0, 0]
    
    num_frames = 35
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for t in range(num_frames):
        alpha = t / float(num_frames - 1)
        
        frame = img_rgb.copy()
        
        for i, cx in enumerate(centers):
            init_st = initial_states[i]
            targ_st = target_states[i]
            
            if init_st == 0 and targ_st == 1:
                # Fade from OFF to ON
                current_patch = (1.0 - alpha) * off_patch + alpha * on_patch
            elif init_st == 1 and targ_st == 0:
                # Fade from ON to OFF
                current_patch = (1.0 - alpha) * on_patch + alpha * off_patch
            elif init_st == 0 and targ_st == 0:
                # Remains OFF
                current_patch = off_patch
            elif init_st == 1 and targ_st == 1:
                # Remains ON
                current_patch = on_patch
            
            current_patch = np.clip(current_patch, 0, 255).astype(np.uint8)
            
            # Place the patch back into the frame
            frame[y_c - patch_size : y_c + patch_size, cx - patch_size : cx + patch_size] = current_patch
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == "__main__":
    main()
