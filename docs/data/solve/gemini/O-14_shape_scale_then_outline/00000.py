import cv2
import numpy as np
import imageio
import os

# Load first frame
img = cv2.imread('/app/first_frame.png')

# Extract test shape mask
shape4 = img[634:634+145, 106:106+135]
mask4 = np.any(shape4 != [255, 255, 255], axis=-1).astype(np.uint8)

# Target middle size
w_mid, h_mid = 114, 122

frames = []

for frame_idx in range(16):
    # Start with fresh copy of first frame
    frame = img.copy()
    
    if frame_idx == 0:
        pass # keep frame as is
        
    elif frame_idx >= 1 and frame_idx <= 7:
        # Phase 1: Middle slot scaling down
        progress = frame_idx / 7.0
        w_cur = int(round(135 - (135 - w_mid) * progress))
        h_cur = int(round(145 - (145 - h_mid) * progress))
        
        # Resize mask4
        mask_cur = cv2.resize(mask4, (w_cur, h_cur), interpolation=cv2.INTER_NEAREST)
        
        # Paste into middle slot
        x_center, y_center = 480.5, 706.5
        x_paste = int(round(x_center - w_cur / 2.0))
        y_paste = int(round(y_center - h_cur / 2.0))
        
        # Draw the mask
        for y in range(h_cur):
            for x in range(w_cur):
                if mask_cur[y, x]:
                    frame[y_paste + y, x_paste + x] = [66, 191, 170]
                    
    elif frame_idx >= 8:
        # Phase 2: Middle slot stays at final size
        mask_mid = cv2.resize(mask4, (w_mid, h_mid), interpolation=cv2.INTER_NEAREST)
        
        # Draw middle slot
        x_paste_mid = int(round(480.5 - w_mid / 2.0))
        y_paste_mid = int(round(706.5 - h_mid / 2.0))
        for y in range(h_mid):
            for x in range(w_mid):
                if mask_mid[y, x]:
                    frame[y_paste_mid + y, x_paste_mid + x] = [66, 191, 170]
                    
        # Right slot animates outline
        pad_size = 5
        padded_mask = np.pad(mask_mid, pad_size, mode='constant')
        kernel = np.ones((7,7), np.uint8)
        mask_out = cv2.morphologyEx(padded_mask, cv2.MORPH_GRADIENT, kernel)
        inner_region = cv2.erode(padded_mask, kernel)
        
        progress = (frame_idx - 8) / 7.0 # 0.0 to 1.0
        
        # Interpolate color for inner region
        color_start = np.array([66, 191, 170], dtype=float)
        color_end = np.array([255, 255, 255], dtype=float)
        cur_color = color_start + (color_end - color_start) * progress
        cur_color = np.round(cur_color).astype(np.uint8)
        
        # Paste right slot
        h_pad, w_pad = padded_mask.shape
        x_paste_right = int(round(787.5 - w_pad / 2.0))
        y_paste_right = int(round(706.5 - h_pad / 2.0))
        
        for y in range(h_pad):
            for x in range(w_pad):
                if mask_out[y, x]:
                    frame[y_paste_right + y, x_paste_right + x] = [66, 191, 170]
                elif inner_region[y, x]:
                    frame[y_paste_right + y, x_paste_right + x] = cur_color
                    
    # Convert BGR to RGB for imageio
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

os.makedirs('/app/output', exist_ok=True)
imageio.mimwrite('/app/output/video.mp4', frames, fps=16, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
print("Done")
