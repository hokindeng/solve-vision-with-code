import cv2
import numpy as np
import imageio
import os

def create_video():
    # 1. Read the first frame
    first_frame_path = '/app/first_frame.png'
    image = cv2.imread(first_frame_path)
    if image is None:
        raise ValueError("Could not read first_frame.png")
        
    # We will work with RGB for video writer
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Identify the hollow points
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bg_color = image[0, 0]
    hollow_points = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        center_color = image[cy, cx]
        
        # If center is same as background, it's hollow
        if np.allclose(center_color, bg_color, atol=10):
            hollow_points.append((cx, cy))
            
    # Sort hollow points left to right
    hollow_points.sort(key=lambda p: p[0])
    
    # 3. Create frames
    frames = []
    fps = 16
    total_frames = 80
    
    # Initial hold
    hold_start = 10
    for _ in range(hold_start):
        frames.append(image_rgb.copy())
        
    # Animation parameters
    frames_per_circle = 10
    radius = 75
    thickness = 6
    red_color = (255, 0, 0) # RGB
    
    current_image = image_rgb.copy()
    
    for pt in hollow_points:
        for i in range(1, frames_per_circle + 1):
            frame = current_image.copy()
            angle = int((i / frames_per_circle) * 360)
            cv2.ellipse(frame, pt, (radius, radius), 0, -90, -90 + angle, red_color, thickness, cv2.LINE_AA)
            frames.append(frame)
        # Update current_image with the fully drawn circle
        cv2.ellipse(current_image, pt, (radius, radius), 0, -90, -90 + 360, red_color, thickness, cv2.LINE_AA)
        
    # Final hold
    remaining_frames = total_frames - len(frames)
    if remaining_frames < 0:
        remaining_frames = 10 # fallback
        
    for _ in range(remaining_frames):
        frames.append(current_image.copy())
        
    # 4. Write video
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'video.mp4')
    
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec='libx264', 
        macro_block_size=None, 
        pixelformat='yuv420p',
        quality=9 # Higher quality to preserve background
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    
    print(f"Video created at {output_path} with {len(frames)} frames.")

if __name__ == '__main__':
    create_video()
