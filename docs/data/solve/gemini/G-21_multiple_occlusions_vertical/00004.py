import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)
    image = cv2.imread('/app/first_frame.png')
    
    # Extract the background by replacing the mask with white
    bg = image.copy()
    mask_color = np.array([209, 209, 209])
    mask_indices = np.all(bg == mask_color, axis=-1)
    bg[mask_indices] = [255, 255, 255]

    num_frames = 58
    fps = 16
    start_y = 30
    end_y = 1024
    
    # The mask width and height
    mask_x = 102
    mask_w = 820
    mask_h = 251

    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')

    for i in range(num_frames):
        y = int(round(start_y + (end_y - start_y) * i / (num_frames - 1)))
        
        frame = bg.copy()
        
        y_start = y
        y_end = y + mask_h
        
        # Clip to frame boundaries
        draw_y_start = max(0, y_start)
        draw_y_end = min(1024, y_end)
        
        if draw_y_start < draw_y_end:
            frame[draw_y_start:draw_y_end, mask_x:mask_x+mask_w] = mask_color
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
