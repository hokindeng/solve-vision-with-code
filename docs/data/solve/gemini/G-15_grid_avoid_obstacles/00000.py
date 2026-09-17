import cv2
import imageio
import numpy as np

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Create clean background
    bg = img.copy()
    bg[793:883, 142:232] = [255, 100, 0] # Fill start cell with blue
    
    # 2. Extract sprite and mask
    sprite = img[814:814+47, 163:163+47].copy()
    mask = (sprite[:, :, 0] != 255) | (sprite[:, :, 1] != 100) | (sprite[:, :, 2] != 0)
    
    # 3. Define path (from BFS)
    path = [
        (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)
    ]
    
    # 4. Generate video
    import os
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    num_frames = 62
    num_segments = len(path) - 1
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        dist = t * num_segments
        seg = int(dist)
        
        if seg >= num_segments:
            seg = num_segments - 1
            frac = 1.0
        else:
            frac = dist - seg
            
        r1, c1 = path[seg]
        r2, c2 = path[seg+1]
        
        y1 = 70 + r1 * 93
        x1 = 70 + c1 * 93
        y2 = 70 + r2 * 93
        x2 = 70 + c2 * 93
        
        y_curr = y1 + (y2 - y1) * frac
        x_curr = x1 + (x2 - x1) * frac
        
        frame = bg.copy()
        
        # Paste sprite
        y_int, x_int = int(round(y_curr)), int(round(x_curr))
        roi = frame[y_int:y_int+47, x_int:x_int+47]
        roi[mask] = sprite[mask]
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
