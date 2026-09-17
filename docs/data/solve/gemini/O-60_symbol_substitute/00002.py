import cv2
import numpy as np
import imageio

def make_video():
    # Read the original image
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Coordinate definitions for Position 4 inner box (the part that changes)
    x_inner = 412
    y_inner = 465
    w_inner = 95
    h_inner = 95
    
    # Old patch to fade out
    old_patch = img_rgb[y_inner:y_inner+h_inner, x_inner:x_inner+w_inner].copy().astype(float)
    white_patch = np.ones_like(old_patch) * 255.0
    
    # Extract reference symbol from the top-right panel
    ref_x = 912
    ref_y = 43
    ref_w = 69
    ref_h = 69
    ref_patch = img_rgb[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w].copy().astype(float)
    
    # Create the new patch to fade in
    new_patch = np.ones_like(old_patch) * 255.0
    
    # Paste reference symbol into the center of new_patch
    pad_y = (h_inner - ref_h) // 2
    pad_x = (w_inner - ref_w) // 2
    new_patch[pad_y:pad_y+ref_h, pad_x:pad_x+ref_w] = ref_patch
    
    frames = []
    num_frames = 52
    
    # Midpoint where the symbol is completely white
    mid_frame = 25
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        if i <= mid_frame:
            # Fade out from old_patch to white_patch
            alpha = i / float(mid_frame)
            current_patch = old_patch * (1.0 - alpha) + white_patch * alpha
        else:
            # Fade in from white_patch to new_patch
            alpha = (i - mid_frame) / float(num_frames - 1 - mid_frame)
            current_patch = white_patch * (1.0 - alpha) + new_patch * alpha
            
        # Overwrite the inner box with the computed patch
        frame[y_inner:y_inner+h_inner, x_inner:x_inner+w_inner] = np.clip(current_patch, 0, 255).astype(np.uint8)
        frames.append(frame)
        
    # Write the output video
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        macro_block_size=None, 
        pixelformat='yuv420p'
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    make_video()
