import imageio
import cv2
import numpy as np
import os

def main():
    os.makedirs('/app/output', exist_ok=True)

    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    # Convert BGR to RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # The agent is located in cell (8,3) initially
    # Agent bounding box is roughly 833:902 (Y), 323:392 (X)
    patch = img_rgb[833:902, 323:392]
    # Agent is orange [255, 165, 0] in RGB
    agent_mask = (patch[:,:,0] == 255) & (patch[:,:,1] == 165) & (patch[:,:,2] == 0)

    # Create a clean background by painting over the initial agent position with green
    bg = img_rgb.copy()
    bg[833:902, 323:392] = [0, 255, 0]

    # Output specifications: H.264, yuv420p, 16 fps
    writer = imageio.get_writer('/app/output/video.mp4', 
                                fps=16, 
                                codec='libx264', 
                                format='FFMPEG', 
                                macro_block_size=None, 
                                pixelformat='yuv420p')

    # Generate 55 frames
    for f in range(55):
        # We hold the first position for a few frames, move smoothly, then hold at the end
        if f < 3:
            d = 0.0
        elif f > 51:
            d = 7.0
        else:
            d = 7.0 * (f - 3) / 48.0
            
        # Path: (8,3) -> (4,3) -> (4,0)
        # Total distance is 7. First 4 units are moving up, next 3 units moving left
        if d <= 4.0:
            r = 8.0 - d
            c = 3.0
        else:
            r = 4.0
            c = 3.0 - (d - 4.0)
            
        # Convert float grid coordinates to pixel coordinates
        # Grid cells are 100x100, plus 2px grid lines (102px stride)
        # Center of cell (r, c) is roughly at y = r*102 + 51, x = c*102 + 51
        cy = int(round(r * 102 + 51))
        cx = int(round(c * 102 + 51))
        
        # Agent bounding box size is 69x69 (radius ~34)
        min_y = cy - 34
        max_y = cy + 34
        min_x = cx - 34
        max_x = cx + 34
        
        frame = bg.copy()
        roi = frame[min_y:max_y+1, min_x:max_x+1]
        
        # Overlay the agent onto the frame
        roi[agent_mask] = [255, 165, 0]
        
        writer.append_data(frame)
        
    writer.close()
    print("Video saved to /app/output/video.mp4")

if __name__ == '__main__':
    main()
