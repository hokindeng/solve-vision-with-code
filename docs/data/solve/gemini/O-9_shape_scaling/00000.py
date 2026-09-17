import cv2
import numpy as np
import imageio
import os

def solve():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    # BGR to RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract C (RGB) - Pentagon
    C_rgb = img_rgb[693:693+137, 184:184+143]
    
    # The scale factor from A to B is exactly 13/15
    target_scale = 13.0 / 15.0
    
    # Target center positions corresponding to the visual grid
    cx = 769.5
    cy = 761.5
    
    total_frames = 60
    anim_frames = 45 # Frames 1 to 45 perform the scaling animation
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for i in range(total_frames):
        if i == 0:
            writer.append_data(img_rgb)
            continue
            
        # Linear scale progress
        progress = (i - 1) / (anim_frames - 1) if anim_frames > 1 else 1.0
        if progress > 1.0:
            progress = 1.0
            
        scale = 1.0 - (1.0 - target_scale) * progress
        
        cur_w = int(round(143 * scale))
        cur_h = int(round(137 * scale))
        
        # Resize using INTER_AREA
        scaled_C_aa = cv2.resize(C_rgb, (cur_w, cur_h), interpolation=cv2.INTER_AREA)
        
        # Make crisp to match the original style (3 colors: white, yellow, black)
        r, g, b = cv2.split(scaled_C_aa)
        
        white_mask = (r > 127) & (g > 127) & (b > 127)
        yellow_mask = (r > 127) & (g > 127) & (b <= 127)
        
        scaled_C = np.zeros_like(scaled_C_aa)
        scaled_C[white_mask] = [255, 255, 255]
        scaled_C[yellow_mask] = [255, 255, 0] # RGB yellow
        scaled_C[~(white_mask | yellow_mask)] = [0, 0, 0]
        
        frame = img_rgb.copy()
        
        # Erase the question mark (fill with white)
        frame[740:795, 750:790] = 255
        
        # Paste scaled C onto the frame
        tx = int(round(cx - cur_w / 2.0))
        ty = int(round(cy - cur_h / 2.0))
        frame[ty:ty+cur_h, tx:tx+cur_w] = scaled_C
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
