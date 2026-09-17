import cv2
import numpy as np
import imageio

def make_video():
    img = cv2.imread('/app/first_frame.png')
    
    # BGR colors
    color_red = np.array([50, 50, 255], dtype=np.float32)
    color_dark_blue = np.array([128, 0, 0], dtype=np.float32)
    color_light_green = np.array([152, 251, 152], dtype=np.float32)
    color_cyan = np.array([220, 220, 50], dtype=np.float32)
    
    # Masks
    mask_dark_blue = cv2.inRange(img, np.array([128, 0, 0]), np.array([128, 0, 0])) > 0
    mask_light_green = cv2.inRange(img, np.array([152, 251, 152]), np.array([152, 251, 152])) > 0
    mask_cyan = cv2.inRange(img, np.array([220, 220, 50]), np.array([220, 220, 50])) > 0
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    total_frames = 84
    
    for i in range(total_frames):
        frame = img.copy().astype(np.float32)
        
        # Timeline
        # Dark Blue -> Red: frames 16 to 25
        if i >= 16:
            progress = min((i - 16) / 9.0, 1.0)
            current_color = color_dark_blue * (1 - progress) + color_red * progress
            frame[mask_dark_blue] = current_color
            
        # Light Green -> Red: frames 41 to 50
        if i >= 41:
            progress = min((i - 41) / 9.0, 1.0)
            current_color = color_light_green * (1 - progress) + color_red * progress
            frame[mask_light_green] = current_color
            
        # Cyan -> Red: frames 66 to 75
        if i >= 66:
            progress = min((i - 66) / 9.0, 1.0)
            current_color = color_cyan * (1 - progress) + color_red * progress
            frame[mask_cyan] = current_color
            
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    make_video()
