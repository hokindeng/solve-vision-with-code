import cv2
import numpy as np
import imageio

def generate_video():
    img_orig = cv2.imread('/app/first_frame.png')
    
    # Tube boundaries (x_left, x_right_inclusive)
    tubes = [
        (223, 312),
        (457, 546),
        (691, 780)
    ]
    
    initial_y = [509, 339, 284]
    final_y = sum(initial_y) / 3.0
    
    k = 2.58
    fps = 16
    num_frames = 53
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    for f in range(num_frames):
        t = f / fps
        frame = img_orig.copy()
        
        for i, (x_start, x_end) in enumerate(tubes):
            current_y_exact = final_y + (initial_y[i] - final_y) * np.exp(-k * t)
            current_y = int(round(current_y_exact))
            
            # If the liquid level went UP (Y decreased)
            if current_y < initial_y[i]:
                # Draw yellow from current_y to initial_y[i]
                # yellow color is BGR (75, 227, 255)
                frame[current_y:initial_y[i], x_start:x_end+1] = (75, 227, 255)
                
            # If the liquid level went DOWN (Y increased)
            elif current_y > initial_y[i]:
                # Draw white from initial_y[i] to current_y
                frame[initial_y[i]:current_y, x_start:x_end+1] = (255, 255, 255)
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
