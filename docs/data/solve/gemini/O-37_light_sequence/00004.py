import cv2
import imageio
import numpy as np
import os

def solve():
    # Load the first frame
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Coordinates of the 10 lights
    light_centers = [
        (142, 512), (224, 512), (306, 512), (388, 512), (470, 512),
        (552, 512), (634, 512), (716, 512), (798, 512), (880, 512)
    ]
    
    # Initial and target states (1 = ON, 0 = OFF)
    initial_states = [1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    target_states = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # Extract reference patches for ON and OFF from the first frame
    generic_on_patch = first_frame_rgb[512-40:512+41, 142-40:142+41].copy()
    generic_off_patch = first_frame_rgb[512-40:512+41, 306-40:306+41].copy()
    
    # Initialize video writer
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p'
    )
    
    num_frames = 35
    for i in range(num_frames):
        # Linear interpolation factor
        t = i / (num_frames - 1)
        
        # Start with a pristine copy of the first frame
        frame = first_frame_rgb.copy()
        
        for j, (cx, cy) in enumerate(light_centers):
            init_s = initial_states[j]
            targ_s = target_states[j]
            
            # If the state doesn't change, we leave the frame as is
            if init_s == targ_s:
                continue
            
            # For smoothly changing the state, extract the EXACT initial patch
            # to ensure the first frame of the video perfectly matches the source.
            initial_patch = first_frame_rgb[cy-40:cy+41, cx-40:cx+41].copy()
            
            # Determine the target patch based on target state
            if targ_s == 1:
                target_patch = generic_on_patch
            else:
                target_patch = generic_off_patch
                
            # Linearly blend the pixel values
            blended_patch = (initial_patch.astype(np.float32) * (1 - t) + 
                             target_patch.astype(np.float32) * t)
            blended_patch = np.clip(blended_patch, 0, 255).astype(np.uint8)
            
            # Place the blended patch back into the image
            frame[cy-40:cy+41, cx-40:cx+41] = blended_patch
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
