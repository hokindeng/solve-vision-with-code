import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)

def draw_minus(patch, x1, y1, width, height, thickness, color):
    patch[y1:y1+height, x1:x1+width] = color
    patch[y1+thickness:y1+height-thickness, x1+thickness:x1+width-thickness] = 255

def main():
    first_frame_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    original_img = cv2.imread(first_frame_path)
    if original_img is None:
        raise ValueError("Could not read first frame")
        
    color = (153, 91, 30) # Blue color in BGR
    
    writer = imageio.get_writer(output_path, fps=16, codec='libx264', pixelformat='yuv420p')
    
    for frame_idx in range(64):
        frame = original_img.copy()
        
        # 1. Fade out the '?' marks
        if frame_idx == 0:
            alpha_q = 0.0
        elif frame_idx <= 8:
            alpha_q = frame_idx / 8.0
        else:
            alpha_q = 1.0
            
        if alpha_q > 0:
            # Erase E '?'
            roi_E = frame[650:715, 450:510]
            white = np.ones_like(roi_E) * 255
            frame[650:715, 450:510] = cv2.addWeighted(roi_E, 1 - alpha_q, white, alpha_q, 0)
            
            # Erase F '?'
            roi_F = frame[650:715, 740:800]
            white2 = np.ones_like(roi_F) * 255
            frame[650:715, 740:800] = cv2.addWeighted(roi_F, 1 - alpha_q, white2, alpha_q, 0)

        # 2. Draw E
        if frame_idx >= 1:
            if frame_idx <= 8:
                alpha_e = frame_idx / 8.0
                patch = np.ones((100, 200, 3), dtype=np.uint8) * 255
                draw_minus(patch, 15, 21, 163, 43, 2, color)
                roi = frame[640:740, 380:580]
                frame[640:740, 380:580] = cv2.addWeighted(roi, 1 - alpha_e, patch, alpha_e, 0)
                
            elif frame_idx <= 24:
                morph = (frame_idx - 8) / 16.0
                patch_start = np.ones((100, 200, 3), dtype=np.uint8) * 255
                draw_minus(patch_start, 15, 21, 163, 43, 2, color)
                
                patch_end = np.ones((100, 200, 3), dtype=np.uint8) * 255
                draw_minus(patch_end, 14, 20, 164, 44, 3, color)
                
                patch = cv2.addWeighted(patch_start, 1 - morph, patch_end, morph, 0)
                roi = frame[640:740, 380:580]
                frame[640:740, 380:580] = cv2.addWeighted(roi, 0, patch, 1.0, 0)
                
            else:
                patch = np.ones((100, 200, 3), dtype=np.uint8) * 255
                draw_minus(patch, 14, 20, 164, 44, 3, color)
                roi = frame[640:740, 380:580]
                frame[640:740, 380:580] = cv2.addWeighted(roi, 0, patch, 1.0, 0)

        # 3. Draw F
        if frame_idx >= 25:
            if frame_idx <= 32:
                alpha_f = (frame_idx - 25) / 7.0
                patch = np.ones((100, 200, 3), dtype=np.uint8) * 255
                draw_minus(patch, 12, 20, 164, 44, 3, color)
                roi = frame[640:740, 670:870]
                frame[640:740, 670:870] = cv2.addWeighted(roi, 1 - alpha_f, patch, alpha_f, 0)
                
            elif frame_idx <= 56:
                progress = (frame_idx - 32) / 24.0
                y_offset = int(round(60.0 * ease_in_out(progress)))
                
                patch_large = np.ones((160, 200, 3), dtype=np.uint8) * 255
                draw_minus(patch_large, 12, 20 + y_offset, 164, 44, 3, color)
                
                # Expand F roi slightly more just in case
                roi = frame[640:800, 670:870]
                frame[640:800, 670:870] = patch_large
                
            else:
                patch_large = np.ones((160, 200, 3), dtype=np.uint8) * 255
                draw_minus(patch_large, 12, 80, 164, 44, 3, color)
                frame[640:800, 670:870] = patch_large
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
