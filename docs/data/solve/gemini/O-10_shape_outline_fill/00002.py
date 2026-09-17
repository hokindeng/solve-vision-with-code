import cv2
import numpy as np
import imageio
import os
import warnings

warnings.filterwarnings("ignore")

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # target bounding box to clear
    x_start, x_end = 760, 970
    y_start, y_end = 665, 875
    
    # Precise Square parameters from C's center + 704 x offset
    left = 782
    right = 946
    top = 687
    bottom = 849
    color = [229, 114, 152] # BGR for purple
    
    perimeter_len = (right - left) * 2 + (bottom - top) * 2
    
    # Generate dashed square map
    square_mask = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32)
    distances = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32) - 1
    
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            is_top = (top <= y <= top + 3)
            is_bot = (bottom - 3 <= y <= bottom)
            is_left = (left <= x <= left + 3)
            is_right = (right - 3 <= x <= right)
            
            if is_top or is_bot or is_left or is_right:
                # continuous distance along perimeter
                if y < top + 4 and x < right - 3:
                    d = x - left
                elif x >= right - 3 and y < bottom - 3:
                    d = (right - left) + (y - top)
                elif y >= bottom - 3 and x > left + 3:
                    d = (right - left) + (bottom - top) + (right - x)
                else:
                    d = (right - left) + (bottom - top) + (right - left) + (bottom - y)
                
                distances[y - y_start, x - x_start] = d
                # 3 pixels on, 1 pixel off
                if (d % 4) < 3:
                    square_mask[y - y_start, x - x_start] = 1.0

    os.makedirs('/app/output', exist_ok=True)
    out = imageio.get_writer('/app/output/video.mp4', fps=16, macro_block_size=1)
    
    for frame_idx in range(60):
        frame = img.copy()
        
        # 1. Fade out the question mark area
        fade_out_frames = 15
        if frame_idx < fade_out_frames:
            alpha = frame_idx / float(fade_out_frames)
        else:
            alpha = 1.0
            
        roi = frame[y_start:y_end, x_start:x_end].copy()
        
        white_bg = np.ones_like(roi) * 255
        blended_bg = cv2.addWeighted(roi, 1.0 - alpha, white_bg, alpha, 0)
        
        # 2. Draw the dashed square
        if frame_idx >= 10:
            # frames 10 to 50
            draw_progress = (frame_idx - 10) / 40.0
            draw_progress = min(max(draw_progress, 0.0), 1.0)
            
            max_d = perimeter_len * draw_progress
            
            for y in range(square_mask.shape[0]):
                for x in range(square_mask.shape[1]):
                    if square_mask[y, x] > 0 and distances[y, x] <= max_d:
                        blended_bg[y, x] = color
                        
        frame[y_start:y_end, x_start:x_end] = blended_bg
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append_data(frame_rgb)
        
    out.close()

if __name__ == '__main__':
    create_video()
