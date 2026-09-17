import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    
    color1 = np.array([0, 128, 128])
    mask1 = cv2.inRange(img, color1, color1)
    color1_rgb = np.array([128, 128, 0])
    
    color2 = np.array([48, 130, 245])
    mask2 = cv2.inRange(img, color2, color2)
    color2_rgb = np.array([245, 130, 48])
    
    y1, x1 = np.where(mask1 > 0)
    center1_x, center1_y = 437, 307
    
    rel_x1 = []
    rel_y1 = []
    for x, y in zip(x1, y1):
        rel_x1.append(x - center1_x)
        rel_y1.append(y - center1_y)
        
        sym_y = 2 * center1_y - y
        if sym_y < 0:
            rel_x1.append(x - center1_x)
            rel_y1.append(sym_y - center1_y)
            
    rel_x1 = np.array(rel_x1)
    rel_y1 = np.array(rel_y1)
    
    y2, x2 = np.where(mask2 > 0)
    center2_x, center2_y = 910, 716
    rel_x2 = x2 - center2_x
    rel_y2 = y2 - center2_y
    
    target_x, target_y = 512, 512
    num_frames = 40
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for t in range(num_frames):
        frame = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        
        progress = t / (num_frames - 1)
        
        cx1 = int(round(center1_x + (target_x - center1_x) * progress))
        cy1 = int(round(center1_y + (target_y - center1_y) * progress))
        
        cx2 = int(round(center2_x + (target_x - center2_x) * progress))
        cy2 = int(round(center2_y + (target_y - center2_y) * progress))
        
        # Draw Circle 1
        x1_draw = rel_x1 + cx1
        y1_draw = rel_y1 + cy1
        valid1 = (x1_draw >= 0) & (x1_draw < 1024) & (y1_draw >= 0) & (y1_draw < 1024)
        frame[y1_draw[valid1], x1_draw[valid1]] = color1_rgb
        
        # Draw Circle 2
        x2_draw = rel_x2 + cx2
        y2_draw = rel_y2 + cy2
        valid2 = (x2_draw >= 0) & (x2_draw < 1024) & (y2_draw >= 0) & (y2_draw < 1024)
        frame[y2_draw[valid2], x2_draw[valid2]] = color2_rgb
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
