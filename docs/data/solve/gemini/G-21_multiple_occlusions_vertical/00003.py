import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask_color = np.array([209, 209, 209])
    
    # Find mask bounding box
    mask_pixels = np.all(img == mask_color, axis=-1)
    y_coords, x_coords = np.where(mask_pixels)
    min_y, max_y = y_coords.min(), y_coords.max()
    min_x, max_x = x_coords.min(), x_coords.max()
    
    mask = img[min_y:max_y+1, min_x:max_x+1].copy()
    
    # Clean background
    bg = img.copy()
    bg[min_y:max_y+1, min_x:max_x+1] = [255, 255, 255]
    
    num_frames = 58
    
    start_y = min_y
    end_y = 1024
    
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, format='FFMPEG', codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    
    for i in range(num_frames):
        frame = bg.copy()
        
        curr_y = int(start_y + (end_y - start_y) * i / (num_frames - 1))
        
        h, w = mask.shape[:2]
        
        y1 = curr_y
        y2 = curr_y + h
        x1 = min_x
        x2 = min_x + w
        
        m_y1 = 0
        m_y2 = h
        
        if y1 < 0:
            m_y1 = -y1
            y1 = 0
        if y2 > frame.shape[0]:
            m_y2 = h - (y2 - frame.shape[0])
            y2 = frame.shape[0]
            
        if y1 < y2:
            frame[y1:y2, x1:x2] = mask[m_y1:m_y2, :]
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == "__main__":
    main()
