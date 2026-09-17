import cv2
import numpy as np
import imageio
import os

def generate_video():
    os.makedirs('/app/output', exist_ok=True)
    
    # Read first frame
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Target inner box at position 3 (y: 465 to 559, x: 413 to 507)
    # The patch size is 95x95
    old_patch = first_frame_rgb[465:560, 413:508].astype(np.float32)
    
    # Diamond from reference panel (y: 43 to 112, x: 912 to 981)
    diamond = first_frame_rgb[43:112, 912:981].astype(np.float32)
    
    # Create pure white patch
    white_patch = np.ones((95, 95, 3), dtype=np.float32) * 255.0
    
    # Create new patch (diamond pasted into white patch)
    new_patch = white_patch.copy()
    new_patch[13:82, 13:82] = diamond
    
    # Video properties
    num_frames = 52
    fps = 16
    output_path = '/app/output/video.mp4'
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for f in range(num_frames):
        t = f / (num_frames - 1)  # 0.0 to 1.0
        
        if t <= 0.5:
            # First half: fade out old symbol to white
            weight_old = 1.0 - (t / 0.5)
            current_patch = weight_old * old_patch + (1.0 - weight_old) * white_patch
        else:
            # Second half: fade in new symbol from white
            weight_new = (t - 0.5) / 0.5
            current_patch = weight_new * new_patch + (1.0 - weight_new) * white_patch
            
        current_patch_uint8 = np.clip(np.round(current_patch), 0, 255).astype(np.uint8)
        
        frame = first_frame_rgb.copy()
        frame[465:560, 413:508] = current_patch_uint8
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
