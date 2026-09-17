import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)
    first_frame = cv2.imread('/app/first_frame.png')

    c1 = np.array([204, 207, 223], dtype=np.float64)
    c2 = np.array([39, 232, 220], dtype=np.float64)
    
    mixed_rgb = np.round(c1 * c2 / 255.0).astype(np.uint8)
    mixed_bgr = mixed_rgb[::-1]

    frames = []
    total_frames = 44

    interior_x_start = 395
    interior_x_end = 630
    interior_y_start = 396
    interior_y_end = 629

    for i in range(total_frames):
        frame = first_frame.copy()
        progress = i / (total_frames - 1)
        
        fill_h = int(round(progress * (interior_y_end - interior_y_start)))
        
        if fill_h > 0:
            y_start = interior_y_end - fill_h
            frame[y_start:interior_y_end, interior_x_start:interior_x_end] = mixed_bgr
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    imageio.mimwrite(
        '/app/output/video.mp4', 
        frames, 
        fps=16, 
        macro_block_size=1, 
        codec='libx264', 
        pixelformat='yuv420p'
    )

if __name__ == '__main__':
    main()
