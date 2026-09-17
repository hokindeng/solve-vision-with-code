import numpy as np
from PIL import Image
import imageio
import os

def main():
    img = Image.open('/app/first_frame.png').convert('RGB')
    arr = np.array(img)
    
    # 1. Extract Coral shape from Slot 4
    # Slot 4 inner bounding box
    slot4_inner = arr[465:560, 570:665]
    diff = np.any(slot4_inner != [255, 255, 255], axis=-1)
    local_y, local_x = np.where(diff)
    global_y = local_y + 465
    global_x = local_x + 570
    
    # Create background by erasing the coral shape
    bg = arr.copy()
    bg[global_y, global_x] = [255, 255, 255]
    
    # 2. Extract Blue Diamond from reference panel
    ref_inner = arr[19:137, 888:1006]
    diamond_diff = np.any(ref_inner != [255, 255, 255], axis=-1)
    dy, dx = np.where(diamond_diff)
    diamond_y_min, diamond_y_max = dy.min(), dy.max()
    diamond_x_min, diamond_x_max = dx.min(), dx.max()
    
    diamond_mask = diamond_diff[diamond_y_min:diamond_y_max+1, diamond_x_min:diamond_x_max+1]
    
    frames = []
    
    # Frame 0: Initial state
    frames.append(arr.copy())
    
    # Phase 1: Slide coral shape right by 105 pixels (10 frames)
    for t in range(1, 11):
        shift_x = int(105 * t / 10)
        frame = bg.copy()
        frame[global_y, global_x + shift_x] = arr[global_y, global_x]
        frames.append(frame)
        
    # Phase 2: Fade in Blue Diamond (8 frames)
    # Target position for blue diamond
    # Inner slot 4 is 95x95, diamond is 78x78. Offset is 8
    target_x = 570 + 8 # 578
    target_y = 465 + 8 # 473
    y_start = target_y - 100 # 373
    
    for t in range(1, 9):
        alpha = t / 8.0
        frame = bg.copy()
        # Coral shape is at final shifted position
        frame[global_y, global_x + 105] = arr[global_y, global_x]
        
        # Blend blue diamond
        for r in range(78):
            for c in range(78):
                if diamond_mask[r, c]:
                    old_pixel = frame[y_start + r, target_x + c].astype(float)
                    new_pixel = np.array([0, 0, 255], dtype=float)
                    blended = old_pixel * (1.0 - alpha) + new_pixel * alpha
                    frame[y_start + r, target_x + c] = blended.astype(np.uint8)
        frames.append(frame)
        
    # Phase 3: Slide down Blue Diamond (13 frames)
    for t in range(1, 14):
        progress = t / 13.0
        current_y = int(y_start + (target_y - y_start) * progress)
        frame = bg.copy()
        # Coral shape is at final shifted position
        frame[global_y, global_x + 105] = arr[global_y, global_x]
        
        # Draw blue diamond
        for r in range(78):
            for c in range(78):
                if diamond_mask[r, c]:
                    frame[current_y + r, target_x + c] = [0, 0, 255]
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    
    # Save video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
