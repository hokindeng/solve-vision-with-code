import cv2
import numpy as np
import imageio
import os

def main():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    img = cv2.imread(img_path)
    
    # BGR colors
    GREEN = [0, 255, 0]
    PURPLE = [130, 0, 75]
    ORANGE = [0, 165, 255]
    
    # Extract parts from the original image to reconstruct the background
    b1 = img[618:736, 87:323].copy()
    b2 = img[618:736, 394:630].copy()
    
    # b1 lever is at 89..146
    # b2 lever is at 12..69
    # Create an empty base track by combining the empty parts
    empty_base = b2.copy()
    empty_base[:, 12:89] = b1[:, 12:89]
    
    # The lever itself is a 57-pixel wide patch
    lever_patch = b1[:, 89:146].copy()
    
    frames = []
    num_frames = 24
    
    # Base X offsets for the three units
    bases_x = [87, 394, 701]
    
    # Lever start and end local X positions (relative to base)
    start_x_levers = [89, 12, 89]
    end_x_levers = [166, 166, 166]
    
    # Light bounding boxes (y_min, y_max, x_min, x_max)
    lights = [
        (202, 297, 157, 252),
        (202, 297, 465, 560),
        (202, 297, 772, 867)
    ]
    
    # Initial colors of the lights (to be used as mask for replacing)
    start_colors = [PURPLE, GREEN, PURPLE]
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    for i in range(num_frames):
        frame = img.copy()
        
        # t goes from 0.0 to 1.0
        t = i / (num_frames - 1) if num_frames > 1 else 1.0
        
        for unit_idx in range(3):
            # Interpolate current lever position
            curr_x = int(start_x_levers[unit_idx] + (end_x_levers[unit_idx] - start_x_levers[unit_idx]) * t)
            
            # Draw the empty base track
            bx = bases_x[unit_idx]
            frame[618:736, bx:bx+236] = empty_base.copy()
            
            # Draw the lever at the new position
            frame[618:736, bx+curr_x : bx+curr_x+57] = lever_patch
            
            # Update light color based on current lever position
            if curr_x < 50:
                target_color = GREEN
            elif curr_x < 127:
                target_color = PURPLE
            else:
                target_color = ORANGE
                
            ly1, ly2, lx1, lx2 = lights[unit_idx]
            light_roi = frame[ly1:ly2, lx1:lx2]
            
            # Mask the original colored part of the light (ignoring black borders and white background)
            old_color = start_colors[unit_idx]
            mask = np.all(light_roi == old_color, axis=-1)
            
            light_roi[mask] = target_color
            frame[ly1:ly2, lx1:lx2] = light_roi
            
        # Convert to RGB for saving with imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite(out_path, frames, fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
    print(f"Video saved to {out_path}")

if __name__ == '__main__':
    main()
