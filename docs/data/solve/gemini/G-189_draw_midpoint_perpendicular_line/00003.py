import cv2
import numpy as np
import imageio

def generate_video():
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Line parameters
    x_center = 239
    y_start = 381
    y_end = 739
    
    num_frames = 50
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = first_frame.copy()
        
        if i > 0:
            # Calculate current y_end for the animated line
            progress = i / (num_frames - 1)
            current_y = int(y_start + (y_end - y_start) * progress)
            
            # Draw line on white pixels only to avoid modifying other elements
            for y in range(y_start, current_y + 1):
                for x in range(x_center - 1, x_center + 3): # 4 pixels wide: 238, 239, 240, 241
                    if (frame[y, x] == [255, 255, 255]).all():
                        frame[y, x] = [255, 0, 0] # Red in RGB
                        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
