import cv2
import numpy as np
import imageio
import os

def generate_video():
    os.makedirs('/app/output', exist_ok=True)
    
    # Load the first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # The left pigment is RGB(204, 171, 33)
    # The right pigment is RGB(136, 88, 158)
    # mixed = round(c1 * c2 / 255)
    
    # c1 = (204, 171, 33)
    # c2 = (136, 88, 158)
    # R = round(204 * 136 / 255) = 109
    # G = round(171 * 88 / 255) = 59
    # B = round(33 * 158 / 255) = 20
    # mixed_rgb = (109, 59, 20)
    # mixed_bgr = (20, 59, 109)
    
    target_color_bgr = np.array([20, 59, 109], dtype=float)
    white_color_bgr = np.array([255, 255, 255], dtype=float)
    
    num_frames = 44
    fps = 16
    
    # Inner dimensions of the black-bordered mixing zone
    y1, y2 = 396, 629  # 629 is exclusive when used in slice, but wait: earlier we saw 628 is the last white pixel.
    # Let's double check. 
    # x in [396, 628] inclusive means slice [396:629]
    # y in [396, 628] inclusive means slice [396:629]
    # We will compute the exact bounds dynamically to be safe.
    
    # Find black border
    mask_black = (first_frame == [0, 0, 0]).all(axis=2)
    y_black, x_black = np.where(mask_black)
    
    if len(y_black) > 0 and len(x_black) > 0:
        # Outer bounds of black border
        y_min, y_max = y_black.min(), y_black.max()
        x_min, x_max = x_black.min(), x_black.max()
        
        # We need to find the white area inside this black border
        # Just scan from the center of the black bounding box
        cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
        
        # Find inner bounds
        y_inner_min = cy
        while y_inner_min >= y_min and np.array_equal(first_frame[y_inner_min, cx], [255, 255, 255]):
            y_inner_min -= 1
        y_inner_min += 1
        
        y_inner_max = cy
        while y_inner_max <= y_max and np.array_equal(first_frame[y_inner_max, cx], [255, 255, 255]):
            y_inner_max += 1
            
        x_inner_min = cx
        while x_inner_min >= x_min and np.array_equal(first_frame[cy, x_inner_min], [255, 255, 255]):
            x_inner_min -= 1
        x_inner_min += 1
        
        x_inner_max = cx
        while x_inner_max <= x_max and np.array_equal(first_frame[cy, x_inner_max], [255, 255, 255]):
            x_inner_max += 1
    else:
        # Fallback to hardcoded if black border is somehow not perfectly [0,0,0]
        y_inner_min, y_inner_max = 396, 629
        x_inner_min, x_inner_max = 396, 629

    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = first_frame.copy()
        
        # Progress from 0 to 1
        progress = i / (num_frames - 1) if num_frames > 1 else 1.0
        
        # Fade color
        current_color = white_color_bgr * (1 - progress) + target_color_bgr * progress
        current_color = np.round(current_color).astype(np.uint8)
        
        # Fill inner region
        frame[y_inner_min:y_inner_max, x_inner_min:x_inner_max] = current_color
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
