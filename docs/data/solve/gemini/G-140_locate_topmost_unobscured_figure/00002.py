import cv2
import numpy as np
import imageio
import os

def solve():
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    # The topmost unobscured shape is the Green triangle.
    # Its color in BGR is [25, 255, 25].
    target_color_bgr = np.array([25, 255, 25])
    
    # Create mask for this color
    mask = cv2.inRange(img, target_color_bgr, target_color_bgr)
    
    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Target shape not found")
        
    cnt = max(contours, key=cv2.contourArea)
    
    # Outline color in BGR (Red) -> (0, 0, 255)
    outline_color = (0, 0, 255)
    thickness = 5
    
    # Video parameters
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    fps = 16
    total_frames = 40
    
    # We will use imageio to ensure h264 and yuv420p
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(total_frames):
        frame = img.copy()
        
        # Calculate how many points to draw
        frac = i / (total_frames - 1)
        num_points = int(len(cnt) * frac)
        
        if num_points > 1:
            is_last = (i == total_frames - 1)
            pts = cnt[:num_points]
            cv2.polylines(frame, [pts], isClosed=is_last, color=outline_color, thickness=thickness, lineType=cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
