import cv2
import numpy as np
import imageio
import os

def main():
    # Load first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Create background by removing the agent and replacing with green
    bg = first_frame.copy()
    yellow_mask = cv2.inRange(bg, np.array([0, 255, 255]), np.array([0, 255, 255]))
    # Replace yellow pixels with the exact green color of the start square
    bg[yellow_mask > 0] = [0, 255, 0] 
    
    # Extract agent mask offsets relative to its center
    y_coords, x_coords = np.where(yellow_mask > 0)
    center_y = 51
    center_x = 153
    dy = y_coords - center_y
    dx = x_coords - center_x
    
    # Path coordinates (centers of cells)
    # Start cell (row 0, col 1): 153, 51
    # End cell (row 8, col 5): 561, 867
    # Path: move down to (153, 867), then right to (561, 867)
    
    total_distance = 816 + 408 # 1224 pixels total
    num_frames = 80
    
    os.makedirs('/app/output', exist_ok=True)
    video_writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None,
        ffmpeg_params=['-crf', '17']
    )
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        d = t * total_distance
        
        if d <= 816:
            cx = 153
            cy = 51 + d
        else:
            cx = 153 + (d - 816)
            cy = 867
            
        cx = int(round(cx))
        cy = int(round(cy))
        
        # Create frame from background
        frame = bg.copy()
        
        # Draw agent
        frame[cy + dy, cx + dx] = [0, 255, 255]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_writer.append_data(frame_rgb)
        
    video_writer.close()

if __name__ == '__main__':
    main()
