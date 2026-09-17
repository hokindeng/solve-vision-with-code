import cv2
import numpy as np
import imageio

def get_frame(img_orig, frame_idx):
    if frame_idx == 0:
        return img_orig.copy()
    
    img = img_orig.copy()
    
    # Fill the original Shape 4 area with white
    # Bounding box of Shape 4 is (119, 628) to (228, 737)
    cv2.rectangle(img, (119, 628), (228, 737), (255, 255, 255), -1)
    
    # Calculate interpolated sizes
    if frame_idx <= 7:
        t = frame_idx / 7.0
        size_out = 109 - 22 * t
        size_in_w = 97 - 22 * t
        size_in_h = 95 - 22 * t
    else:
        t = (frame_idx - 8) / 7.0
        size_out = 87 - 4 * t
        size_in_w = 75 + 4 * t
        size_in_h = 73 + 4 * t
        
    center_x = 173.5
    center_y = 682.5
    
    x_out_min = int(round(center_x - size_out / 2))
    y_out_min = int(round(center_y - size_out / 2))
    x_out_max = int(round(center_x + size_out / 2))
    y_out_max = int(round(center_y + size_out / 2))
    
    x_in_min = int(round(center_x - size_in_w / 2))
    y_in_min = int(round(center_y - size_in_h / 2))
    x_in_max = int(round(center_x + size_in_w / 2))
    y_in_max = int(round(center_y + size_in_h / 2))
    
    # Draw outer rectangle
    cv2.rectangle(img, (x_out_min, y_out_min), (x_out_max - 1, y_out_max - 1), (95, 191, 175), -1)
    
    # Draw inner rectangle
    cv2.rectangle(img, (x_in_min, y_in_min), (x_in_max - 1, y_in_max - 1), (255, 255, 255), -1)
    
    # BGR to RGB for imageio
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    img_orig = cv2.imread('/app/first_frame.png')
    
    frames = []
    for i in range(16):
        if i == 0:
            frames.append(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        else:
            frames.append(get_frame(img_orig, i))
            
    # Save video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, quality=9)

if __name__ == '__main__':
    main()
