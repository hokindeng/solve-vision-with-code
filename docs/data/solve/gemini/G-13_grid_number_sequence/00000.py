import cv2
import numpy as np
import imageio
import os

def get_pixel(col, row):
    return (col * 102 + 17, row * 102 + 17)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Create clean background
    clean_bg = img.copy()
    orange_mask = np.all(img == [0, 165, 255], axis=-1)
    
    # Fill orange pixels with green (the color of the start cell)
    clean_bg[orange_mask] = [0, 255, 0]
    
    # Extract agent sprite
    y_idx, x_idx = np.where(orange_mask)
    min_y, max_y = y_idx.min(), y_idx.max()
    min_x, max_x = x_idx.min(), x_idx.max()
    
    agent_sprite = img[min_y:max_y+1, min_x:max_x+1].copy()
    agent_alpha = orange_mask[min_y:max_y+1, min_x:max_x+1]
    
    # Define waypoints (Start -> 1 -> 2 -> 3 -> End)
    # Using L-shapes to avoid passing through other waypoints
    waypoints = [
        (2, 4), # Start
        (2, 8), (1, 8), # Path to 1
        (3, 8), (3, 2), # Path to 2
        (8, 2), (8, 5), # Path to 3
        (1, 5), (1, 4)  # Path to End
    ]
    
    pixel_path = []
    for i in range(len(waypoints) - 1):
        c1, r1 = waypoints[i]
        c2, r2 = waypoints[i+1]
        
        x1, y1 = get_pixel(c1, r1)
        x2, y2 = get_pixel(c2, r2)
        
        if x1 != x2:
            step = 1 if x2 > x1 else -1
            for x in range(x1, x2, step):
                pixel_path.append((x, y1))
        elif y1 != y2:
            step = 1 if y2 > y1 else -1
            for y in range(y1, y2, step):
                pixel_path.append((x1, y))
                
    # Add final position
    c_end, r_end = waypoints[-1]
    pixel_path.append(get_pixel(c_end, r_end))
    
    num_frames = 107
    indices = np.linspace(0, len(pixel_path) - 1, num_frames)
    indices = np.round(indices).astype(int)
    
    os.makedirs('/app/output', exist_ok=True)
    
    # Write video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for idx in indices:
        x, y = pixel_path[idx]
        frame = clean_bg.copy()
        
        # Paste agent
        # The agent sprite size is (69, 69)
        h, w = agent_sprite.shape[:2]
        frame_slice = frame[y:y+h, x:x+w]
        frame_slice[agent_alpha] = agent_sprite[agent_alpha]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
