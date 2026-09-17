import cv2
import numpy as np
import os
import imageio

def generate_video():
    # Read the first frame
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Coordinates of reference panel
    ref = first_frame[18:138, 887:1007]
    
    # Target color: Violet in RGB is [238, 130, 238]
    mask_ref = (ref[:,:,0] == 238) & (ref[:,:,1] == 130) & (ref[:,:,2] == 238)
    ys, xs = np.where(mask_ref)
    
    star_h = ys.max() - ys.min() + 1
    star_w = xs.max() - xs.min() + 1
    
    # Tightly cropped mask of the violet star
    star_mask = mask_ref[ys.min():ys.max()+1, xs.min():xs.max()+1]
    
    # Position to center the star in the 97x97 cell
    start_y = int(48.0 - star_h / 2.0)
    start_x = int(48.0 - star_w / 2.0)
    
    # Coordinates of sym3: y=464:561, x=306:403
    sym3_roi = first_frame[464:464+97, 306:306+97]
    
    # Old color: Yellow in RGB is [255, 255, 0]
    mask_old = (sym3_roi[:,:,0] == 255) & (sym3_roi[:,:,1] == 255) & (sym3_roi[:,:,2] == 0)
    
    num_frames = 52
    
    os.makedirs('/app/output', exist_ok=True)
    
    # Prepare video writer
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p', 
        macro_block_size=None
    )
    
    for i in range(num_frames):
        frame = first_frame.copy()
        roi = frame[464:464+97, 306:306+97]
        
        # We have 52 frames (0 to 51)
        # 25.5 is the exact midpoint for transition where it's purely white
        if i <= 25:
            # Phase 1: Fade out old symbol
            alpha = 1.0 - (i / 25.5)
            old_color = np.array([255, 255, 0], dtype=float)
            white = np.array([255, 255, 255], dtype=float)
            blended = alpha * old_color + (1 - alpha) * white
            roi[mask_old] = np.round(blended).astype(np.uint8)
        else:
            # Phase 2: Fade in new symbol
            # First, erase the old symbol completely
            roi[mask_old] = [255, 255, 255]
            
            alpha = (i - 25.5) / 25.5
            new_color = np.array([238, 130, 238], dtype=float)
            white = np.array([255, 255, 255], dtype=float)
            blended = alpha * new_color + (1 - alpha) * white
            
            # Draw the new symbol with alpha blending
            star_roi = roi[start_y:start_y+star_h, start_x:start_x+star_w]
            star_roi[star_mask] = np.round(blended).astype(np.uint8)
            
        writer.append_data(frame)
        
    writer.close()
    
if __name__ == '__main__':
    generate_video()
