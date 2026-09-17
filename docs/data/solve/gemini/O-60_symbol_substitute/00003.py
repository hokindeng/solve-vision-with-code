import cv2
import numpy as np
import imageio
import os

def main():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
    
    # Convert from BGR to RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Colors (RGB)
    C_old = np.array([255, 255, 0], dtype=float)    # Yellow
    C_new = np.array([75, 0, 130], dtype=float)     # Indigo
    C_bg  = np.array([255, 255, 255], dtype=float)  # White

    # Position 4 box coordinates
    # We found x=517, y=464, w=97, h=97 for the box
    box_x, box_y = 517, 464
    box_w, box_h = 97, 97
    obj4_box = img_rgb[box_y:box_y+box_h, box_x:box_x+box_w]

    # Mask for the old symbol
    old_mask = np.all(obj4_box == C_old, axis=-1)

    # Reference panel for the new symbol
    # We found x=887, y=18, w=120, h=120
    ref_panel = img_rgb[18:18+120, 887:887+120]
    heart_mask_full = np.all(ref_panel == C_new, axis=-1)

    # Find bounding box of the heart in the reference panel
    coords = np.column_stack(np.where(heart_mask_full))
    if coords.size == 0:
        raise ValueError("Could not find the target symbol in the reference panel.")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    heart_mask = heart_mask_full[y_min:y_max+1, x_min:x_max+1]
    h, w = heart_mask.shape

    # Create a centered mask for the new symbol inside the 97x97 box
    new_mask = np.zeros((box_h, box_w), dtype=bool)
    start_x = (box_w - w) // 2
    start_y = (box_h - h) // 2
    new_mask[start_y:start_y+h, start_x:start_x+w] = heart_mask

    # Create a clean box (with only background and border)
    clean_box = obj4_box.copy()
    clean_box[old_mask] = C_bg.astype(np.uint8)

    # Generate the frames
    frames = []
    total_frames = 52
    for i in range(total_frames):
        progress = i / (total_frames - 1)
        current_box = clean_box.copy()
        
        if progress <= 0.5:
            # Fade out phase: old color -> white
            alpha = progress * 2.0
            current_color = (1.0 - alpha) * C_old + alpha * C_bg
            current_box[old_mask] = np.round(current_color).astype(np.uint8)
        else:
            # Fade in phase: white -> new color
            alpha = (progress - 0.5) * 2.0
            current_color = (1.0 - alpha) * C_bg + alpha * C_new
            current_box[new_mask] = np.round(current_color).astype(np.uint8)
            
        frame = img_rgb.copy()
        frame[box_y:box_y+box_h, box_x:box_x+box_w] = current_box
        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
