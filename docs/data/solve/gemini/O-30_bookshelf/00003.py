import cv2
import numpy as np
import imageio
import math
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Book 0 info
    x0_start, y0_start = 754, 371
    w0, h0 = 33, 139
    x0_end, y0_end = 190, 371
    
    # Book 1 info
    x1_start, y1_start = 792, 356
    w1, h1 = 33, 154
    x1_end, y1_end = 304, 356
    
    # Extract book images
    book0_img = img_rgb[y0_start:y0_start+h0, x0_start:x0_start+w0].copy()
    book1_img = img_rgb[y1_start:y1_start+h1, x1_start:x1_start+w1].copy()
    
    # Create clean background
    bg_img = img_rgb.copy()
    bg_img[y0_start:y0_start+h0, x0_start:x0_start+w0] = [255, 255, 255]
    bg_img[y1_start:y1_start+h1, x1_start:x1_start+w1] = [255, 255, 255]
    
    frames = []
    
    total_frames = 24
    
    for f in range(total_frames):
        frame = bg_img.copy()
        
        # Calculate pos for book 0
        if f <= 11:
            t = f / 11.0
            x0 = int(x0_start + (x0_end - x0_start) * t)
            y0 = int(y0_start - 271 * math.sin(math.pi * t))
        else:
            x0 = x0_end
            y0 = y0_end
            
        # Calculate pos for book 1
        if f <= 11:
            x1 = x1_start
            y1 = y1_start
        else:
            t = (f - 11) / 12.0
            x1 = int(x1_start + (x1_end - x1_start) * t)
            y1 = int(y1_start - 256 * math.sin(math.pi * t))
            
        # Paste books. Paste book 1 then book 0 (order doesn't matter unless they overlap)
        frame[y1:y1+h1, x1:x1+w1] = book1_img
        frame[y0:y0+h0, x0:x0+w0] = book0_img
        
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    solve()
