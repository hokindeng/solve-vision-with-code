import cv2
import numpy as np
import imageio

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Agent properties
    y_min, y_max = 814, 860
    x_min, x_max = 349, 395
    
    patch = img[y_min:y_max+1, x_min:x_max+1]
    
    # Background color of the start cell
    bg_color = [255, 100, 0] # BGR
    bg_mask = np.all(patch == bg_color, axis=-1)
    agent_mask = ~bg_mask
    
    # Extract agent RGBA
    agent_rgba = np.zeros((47, 47, 4), dtype=np.float32)
    agent_rgba[..., :3] = patch
    agent_rgba[..., 3] = agent_mask.astype(np.float32)
    
    # Create clean background
    img_clean = img.copy()
    img_clean[y_min:y_max+1, x_min:x_max+1][agent_mask] = bg_color
    
    # Path
    start_y, start_x = 837, 372
    end_y, end_x = 279, 372
    num_frames = 34
    
    frames = []
    
    for i in range(num_frames):
        # Interpolate position
        y_center = start_y + (end_y - start_y) * i / (num_frames - 1)
        x_center = start_x + (end_x - start_x) * i / (num_frames - 1)
        
        y1 = int(round(y_center)) - 23
        x1 = int(round(x_center)) - 23
        
        frame = img_clean.copy().astype(np.float32)
        alpha = agent_rgba[..., 3]
        
        # Alpha compositing
        for c in range(3):
            frame[y1:y1+47, x1:x1+47, c] = (
                alpha * agent_rgba[..., c] +
                (1 - alpha) * frame[y1:y1+47, x1:x1+47, c]
            )
            
        # Convert BGR to RGB for imageio
        frame = frame.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # Write video
    import os
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')
    print('Video generated successfully.')

if __name__ == '__main__':
    create_video()
